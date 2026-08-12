"""TTS module - Chinese text-to-speech using edge-tts or IndexTTS."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from kakure.config import Settings, TTSBackend

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of TTS generation for a single segment."""

    segment_id: int
    audio_path: Path
    duration_ms: int  # Duration in milliseconds


class BaseTTSProcessor(ABC):
    """Abstract base class for TTS processors."""

    @abstractmethod
    def generate_sync(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Generate TTS audio for all translated segments.

        Args:
            segments: List of dicts with 'id', 'translated_text', 'start', 'end'.
            output_dir: Directory to save audio files.

        Returns:
            List of TTSResult with audio paths and durations.
        """
        ...


class EdgeTTSProcessor(BaseTTSProcessor):
    """Generates Chinese speech audio from translated text using edge-tts."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.voice = self.settings.chinese_voice.value
        self.rate = self.settings.tts_rate
        self.volume = self.settings.tts_volume
        self.pitch = self.settings.tts_pitch
        self._temp_dir = Path(tempfile.mkdtemp(prefix="kakure_tts_"))

    async def _generate_segment(self, text: str, output_path: Path) -> Path:
        """Generate TTS audio for a single text segment using edge-tts.

        Args:
            text: Chinese text to synthesize.
            output_path: Path to save the audio file.

        Returns:
            Path to the generated audio file.
        """
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        await communicate.save(str(output_path))
        return output_path

    async def generate(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Generate Chinese TTS audio for all translated segments.

        Args:
            segments: List of dicts with 'id', 'translated_text', 'start', 'end'.
            output_dir: Directory to save audio files. Defaults to temp dir.

        Returns:
            List of TTSResult with audio paths and durations.
        """
        if output_dir is None:
            output_dir = self._temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating TTS for %d segments using edge-tts voice %s", len(segments), self.voice
        )
        results: list[TTSResult] = []

        for seg in segments:
            text = seg["translated_text"]
            if not text or not text.strip():
                logger.warning("Skipping empty segment %d", seg["id"])
                continue

            output_path = output_dir / f"segment_{seg['id']:04d}.mp3"

            # Reuse an already-synthesized segment (checkpoint resume / crash recovery)
            if output_path.exists() and output_path.stat().st_size > 0:
                from pydub import AudioSegment

                audio = AudioSegment.from_mp3(str(output_path))
                duration_ms = len(audio)
                results.append(
                    TTSResult(
                        segment_id=seg["id"],
                        audio_path=output_path,
                        duration_ms=duration_ms,
                    )
                )
                logger.debug("Reusing existing TTS segment %d (%dms)", seg["id"], duration_ms)
                continue

            try:
                await self._generate_segment(text, output_path)

                # Get duration using pydub
                from pydub import AudioSegment

                audio = AudioSegment.from_mp3(str(output_path))
                duration_ms = len(audio)

                results.append(
                    TTSResult(
                        segment_id=seg["id"],
                        audio_path=output_path,
                        duration_ms=duration_ms,
                    )
                )
                logger.debug(
                    "TTS segment %d: %dms, %s",
                    seg["id"],
                    duration_ms,
                    text[:30],
                )
            except Exception as e:
                logger.error("Failed to generate TTS for segment %d: %s", seg["id"], e)
                continue

        logger.info("edge-tts generation complete: %d segments", len(results))
        return results

    def generate_sync(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Synchronous wrapper for generate()."""
        return asyncio.run(self.generate(segments, output_dir))

    @staticmethod
    async def list_chinese_voices() -> list[dict]:
        """List all available Chinese voices from edge-tts."""
        import edge_tts

        voices = await edge_tts.list_voices()
        return [v for v in voices if v["Locale"].startswith("zh-CN")]

    @staticmethod
    def list_chinese_voices_sync() -> list[dict]:
        """Synchronous wrapper for list_chinese_voices()."""
        return asyncio.run(EdgeTTSProcessor.list_chinese_voices())


class IndexTTSProcessor(BaseTTSProcessor):
    """Generates Chinese speech audio using IndexTTS (local GPU model with voice cloning).

    IndexTTS requires a reference audio file for voice cloning and an NVIDIA GPU.
    It outputs WAV files and supports Chinese text with pinyin control.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.reference_audio = self.settings.indextts_reference_audio
        self.model_dir = self.settings.indextts_model_dir
        self.language = self.settings.indextts_language
        self._model = None
        self._temp_dir = Path(tempfile.mkdtemp(prefix="kakure_tts_"))

    @property
    def model(self):
        """Lazy-load the IndexTTS model."""
        if self._model is None:
            from indextts import IndexTTS2

            logger.info("Loading IndexTTS model")
            if self.model_dir:
                self._model = IndexTTS2(model_dir=self.model_dir)
            else:
                # Auto-download from HuggingFace
                self._model = IndexTTS2()
            logger.info("IndexTTS model loaded successfully")
        return self._model

    def generate_sync(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Generate Chinese TTS audio for all translated segments using IndexTTS.

        Args:
            segments: List of dicts with 'id', 'translated_text', 'start', 'end'.
            output_dir: Directory to save audio files. Defaults to temp dir.

        Returns:
            List of TTSResult with audio paths and durations.
        """
        if not self.reference_audio:
            raise ValueError(
                "IndexTTS requires a reference audio file for voice cloning. "
                "Upload one in the web UI or set indextts_reference_audio in kakure.toml."
            )
        if not Path(self.reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")

        if output_dir is None:
            output_dir = self._temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating TTS for %d segments using IndexTTS (ref: %s)",
            len(segments),
            self.reference_audio,
        )
        results: list[TTSResult] = []

        for seg in segments:
            text = seg["translated_text"]
            if not text or not text.strip():
                logger.warning("Skipping empty segment %d", seg["id"])
                continue

            # IndexTTS outputs WAV files
            output_path = output_dir / f"segment_{seg['id']:04d}.wav"

            # Reuse an already-synthesized segment (checkpoint resume / crash recovery)
            if output_path.exists() and output_path.stat().st_size > 0:
                from pydub import AudioSegment

                audio = AudioSegment.from_wav(str(output_path))
                duration_ms = len(audio)
                results.append(
                    TTSResult(
                        segment_id=seg["id"],
                        audio_path=output_path,
                        duration_ms=duration_ms,
                    )
                )
                logger.debug("Reusing existing TTS segment %d (%dms)", seg["id"], duration_ms)
                continue

            try:
                self.model.infer(
                    spk_audio_prompt=str(self.reference_audio),
                    text=text,
                    output_path=str(output_path),
                )

                # Get duration using pydub
                from pydub import AudioSegment

                audio = AudioSegment.from_wav(str(output_path))
                duration_ms = len(audio)

                results.append(
                    TTSResult(
                        segment_id=seg["id"],
                        audio_path=output_path,
                        duration_ms=duration_ms,
                    )
                )
                logger.debug(
                    "IndexTTS segment %d: %dms, %s",
                    seg["id"],
                    duration_ms,
                    text[:30],
                )
            except Exception as e:
                logger.error("Failed to generate TTS for segment %d: %s", seg["id"], e)
                continue

        logger.info("IndexTTS generation complete: %d segments", len(results))
        return results


def create_tts_processor(settings: Settings | None = None) -> BaseTTSProcessor:
    """Factory function to create the appropriate TTS processor based on settings.

    Args:
        settings: Application settings. Uses defaults if None.

    Returns:
        An EdgeTTSProcessor or IndexTTSProcessor instance.
    """
    settings = settings or Settings()

    if settings.tts_backend == TTSBackend.INDEX_TTS:
        logger.info("Using IndexTTS TTS backend")
        return IndexTTSProcessor(settings)
    else:
        logger.info("Using edge-tts TTS backend")
        return EdgeTTSProcessor(settings)
