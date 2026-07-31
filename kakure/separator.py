"""Vocal separator module - uses Demucs to split audio into vocals and background."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from kakure.config import Settings

logger = logging.getLogger(__name__)


def _tensor_to_wav(tensor, path, sample_rate):
    """Save a torch tensor as a 16-bit PCM WAV file using only stdlib."""
    import wave

    import numpy as np

    wav = tensor.detach().cpu().numpy()
    if wav.ndim == 1:
        wav = wav.reshape(1, -1)
    n_channels = wav.shape[0]
    wav = np.clip(wav, -1.0, 1.0)
    wav = (wav * 32767).astype(np.int16)
    wav = wav.T
    with wave.open(str(path), "w") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wav.tobytes())


@dataclass
class SeparatedAudio:
    """Result of vocal/background separation."""

    original: AudioSegment  # Original unmodified audio
    vocals: AudioSegment  # Isolated vocals track
    background: AudioSegment  # Background (drums + bass + other) track


class VocalSeparator:
    """Separates audio into vocals and background using Demucs.

    This allows the mixer to reduce original Japanese vocals while keeping
    background music/sounds, creating a cleaner bilingual overlay where the
    Chinese TTS doesn't compete with the original vocals.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._separator = None

    @property
    def separator(self):
        """Lazy-load the Demucs separator."""
        if self._separator is None:
            import demucs.api

            logger.info(
                "Loading Demucs model '%s' on %s",
                self.settings.demucs_model.value,
                self.settings.demucs_device,
            )
            self._separator = demucs.api.Separator(
                model=self.settings.demucs_model.value,
                device=self.settings.demucs_device,
            )
            logger.info("Demucs model loaded successfully")
        return self._separator

    def separate(self, audio_path: Path | str) -> SeparatedAudio:
        """Separate an audio file into vocals and background.

        Args:
            audio_path: Path to the audio file to separate.

        Returns:
            SeparatedAudio with original, vocals, and background tracks.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Separating vocals from %s using Demucs", audio_path)

        # Use Demucs API to separate
        _, separated = self.separator.separate_audio_file(str(audio_path))

        # Extract vocals and background
        # Demucs returns: {'drums': tensor, 'bass': tensor, 'other': tensor, 'vocals': tensor}
        vocals_tensor = separated.get("vocals")
        if vocals_tensor is None:
            raise ValueError("Demucs did not return a vocals stem")

        # Combine non-vocal stems into background
        background_tensor = None
        for stem_name in ("drums", "bass", "other"):
            stem_tensor = separated.get(stem_name)
            if stem_tensor is not None:
                if background_tensor is None:
                    background_tensor = stem_tensor
                else:
                    background_tensor = background_tensor + stem_tensor

        # If no background stems, create silence

        original_audio = AudioSegment.from_file(str(audio_path))
        sample_rate = self.separator.samplerate

        # Convert tensors to AudioSegments via temp WAV files
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="kakure_demucs_"))

        vocals_path = temp_dir / "vocals.wav"
        _tensor_to_wav(vocals_tensor, vocals_path, sample_rate)
        vocals = AudioSegment.from_wav(str(vocals_path))

        if background_tensor is not None:
            background_path = temp_dir / "background.wav"
            _tensor_to_wav(background_tensor, background_path, sample_rate)
            background = AudioSegment.from_wav(str(background_path))
        else:
            # No background stems — create silence matching original length
            background = AudioSegment.silent(duration=len(original_audio))

        # Match lengths (Demucs may produce slightly different lengths)
        max_len = max(len(original_audio), len(vocals), len(background))
        original_audio = original_audio + AudioSegment.silent(
            duration=max(0, max_len - len(original_audio))
        )
        vocals = vocals + AudioSegment.silent(
            duration=max(0, max_len - len(vocals))
        )
        background = background + AudioSegment.silent(
            duration=max(0, max_len - len(background))
        )

        logger.info(
            "Vocal separation complete: original=%dms, vocals=%dms, background=%dms",
            len(original_audio),
            len(vocals),
            len(background),
        )

        return SeparatedAudio(
            original=original_audio,
            vocals=vocals,
            background=background,
        )
