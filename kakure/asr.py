"""ASR module - Japanese speech recognition using faster-whisper or kotoba-whisper."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from kakure.config import ASRBackend, Settings

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A transcribed audio segment with timing information."""

    id: int
    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start


@dataclass
class Word:
    """A transcribed word with timing information."""

    start: float
    end: float
    word: str
    probability: float


@dataclass
class TranscriptionResult:
    """Result of ASR transcription."""

    segments: list[Segment]
    language: str
    language_probability: float
    duration: float


class BaseASRProcessor(ABC):
    """Abstract base class for ASR processors."""

    @abstractmethod
    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe a Japanese audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        ...


class ASRProcessor(BaseASRProcessor):
    """Processes audio files using faster-whisper for Japanese speech recognition."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' on %s with compute_type=%s",
                self.settings.whisper_model.value,
                self.settings.whisper_device,
                self.settings.whisper_compute_type,
            )
            self._model = WhisperModel(
                self.settings.whisper_model.value,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
            logger.info("faster-whisper model loaded successfully")
        return self._model

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe a Japanese audio file using faster-whisper.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing %s with faster-whisper", audio_path)
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.settings.whisper_language,
            beam_size=self.settings.whisper_beam_size,
            vad_filter=self.settings.whisper_vad_filter,
            word_timestamps=True,
        )

        # Convert iterator to list of our Segment objects
        segments: list[Segment] = []
        for idx, seg in enumerate(segments_iter):
            words = [
                Word(
                    start=w.start,
                    end=w.end,
                    word=w.word.strip(),
                    probability=w.probability,
                )
                for w in (seg.words or [])
            ]
            segment = Segment(
                id=idx,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
            )
            segments.append(segment)
            logger.debug(
                "Segment %d: [%.2fs -> %.2fs] %s",
                segment.id,
                segment.start,
                segment.end,
                segment.text,
            )

        total_duration = segments[-1].end if segments else 0.0
        result = TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=total_duration,
        )

        logger.info(
            "faster-whisper transcription complete: %d segments, language=%s (%.2f%% confidence)",
            len(segments),
            info.language,
            info.language_probability * 100,
        )
        return result


class KotobaWhisperProcessor(BaseASRProcessor):
    """Processes audio files using kotoba-whisper (HuggingFace Transformers) for Japanese ASR.

    kotoba-whisper is a Japanese-optimized Whisper model that provides better CER
    than generic Whisper models. It uses the HuggingFace Transformers pipeline API.

    Note: kotoba-whisper only provides segment-level timestamps, not word-level.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the kotoba-whisper pipeline."""
        if self._pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline

            model_id = self.settings.kotoba_whisper_model.value
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_kwargs = (
                {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
            )

            logger.info(
                "Loading kotoba-whisper model '%s' on %s",
                model_id,
                device,
            )
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs,
                batch_size=16,
                trust_remote_code=True,
            )
            logger.info("kotoba-whisper model loaded successfully")
        return self._pipeline

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe a Japanese audio file using kotoba-whisper.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with segments and metadata.
            Note: word-level timestamps are not available (words list will be empty).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing %s with kotoba-whisper", audio_path)

        generate_kwargs = {"language": self.settings.whisper_language, "task": "transcribe"}

        result = self.pipeline(
            str(audio_path),
            chunk_length_s=self.settings.kotoba_whisper_chunk_length_s,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        # Convert kotoba-whisper chunks to our Segment objects
        segments: list[Segment] = []
        chunks = result.get("chunks", [])

        if chunks:
            for idx, chunk in enumerate(chunks):
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                start = timestamp[0] if timestamp[0] is not None else 0.0
                end = timestamp[1] if timestamp[1] is not None else start
                text = chunk.get("text", "").strip()

                if not text:
                    continue

                segment = Segment(
                    id=idx,
                    start=start,
                    end=end,
                    text=text,
                    words=[],  # kotoba-whisper doesn't provide word-level timestamps
                )
                segments.append(segment)
                logger.debug(
                    "Segment %d: [%.2fs -> %.2fs] %s",
                    segment.id,
                    segment.start,
                    segment.end,
                    segment.text,
                )
        else:
            # Fallback: single segment with full text
            text = result.get("text", "").strip()
            if text:
                segments.append(Segment(id=0, start=0.0, end=0.0, text=text))

        total_duration = segments[-1].end if segments else 0.0
        transcription = TranscriptionResult(
            segments=segments,
            language="ja",
            language_probability=1.0,  # kotoba-whisper is Japanese-specific
            duration=total_duration,
        )

        logger.info(
            "kotoba-whisper transcription complete: %d segments",
            len(segments),
        )
        return transcription


def create_asr_processor(settings: Settings | None = None) -> BaseASRProcessor:
    """Factory function to create the appropriate ASR processor based on settings.

    Args:
        settings: Application settings. Uses defaults if None.

    Returns:
        An ASRProcessor or KotobaWhisperProcessor instance.
    """
    settings = settings or Settings()

    if settings.asr_backend == ASRBackend.KOTOBA_WHISPER:
        logger.info("Using kotoba-whisper ASR backend")
        return KotobaWhisperProcessor(settings)
    else:
        logger.info("Using faster-whisper ASR backend")
        return ASRProcessor(settings)
