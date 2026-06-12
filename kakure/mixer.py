"""Audio mixer module - combines original and translated audio tracks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydub import AudioSegment

from kakure.config import MixMode, Settings

if TYPE_CHECKING:
    from kakure.separator import SeparatedAudio

logger = logging.getLogger(__name__)


@dataclass
class MixInput:
    """Input for the audio mixer."""

    original_audio: AudioSegment
    segments: list[dict]  # Each: {id, start, end, original_text, translated_text}
    tts_results: list[dict]  # Each: {segment_id, audio_path, duration_ms}
    separated: SeparatedAudio | None = None  # Optional: vocals/background from Demucs


class AudioMixer:
    """Mixes original Japanese audio with Chinese TTS audio.

    When vocal separation (Demucs) is enabled, the mixer uses the separated
    vocals and background tracks to create a cleaner bilingual mix:
    - Background plays at full volume throughout
    - Original vocals are reduced so Chinese TTS doesn't compete
    - Chinese TTS is overlaid at the configured volume
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def mix(self, mix_input: MixInput) -> AudioSegment:
        """Mix original and translated audio according to the configured mode.

        Args:
            mix_input: MixInput with original audio, segments, and TTS results.

        Returns:
            Mixed AudioSegment.
        """
        mode = self.settings.mix_mode
        logger.info("Mixing audio in %s mode", mode.value)

        if mix_input.separated is not None:
            logger.info("Using separated vocals/background from Demucs")

        if mode == MixMode.DUAL:
            return self._mix_dual(mix_input)
        elif mode == MixMode.OVERLAY:
            return self._mix_overlay(mix_input)
        elif mode == MixMode.SEQUENTIAL:
            return self._mix_sequential(mix_input)
        elif mode == MixMode.WHISPER:
            return self._mix_whisper(mix_input)
        else:
            raise ValueError(f"Unknown mix mode: {mode}")

    def _get_base_audio(self, mix_input: MixInput) -> AudioSegment:
        """Get the base audio track for mixing.

        When vocals are separated, returns background + reduced vocals.
        Otherwise returns the original audio unchanged.
        """
        if mix_input.separated is not None:
            # Mix background at configured volume + reduced vocals
            background = mix_input.separated.background + self.settings.background_volume_db
            vocals = mix_input.separated.vocals + self.settings.vocals_volume_db
            return background.overlay(vocals)
        return mix_input.original_audio

    def _get_vocals_audio(self, mix_input: MixInput) -> AudioSegment | None:
        """Get the vocals track if separated, None otherwise."""
        if mix_input.separated is not None:
            return mix_input.separated.vocals
        return None

    def _load_tts_audio(self, segment_id: int, tts_results: list[dict]) -> AudioSegment | None:
        """Load TTS audio for a segment."""
        for tts in tts_results:
            if tts["segment_id"] == segment_id:
                path = tts["audio_path"]
                if Path(path).exists():
                    return AudioSegment.from_file(str(path))
        return None

    def _mix_dual(self, mix_input: MixInput) -> AudioSegment:
        """Dual mode: Japanese in left channel, Chinese in right channel.

        Creates a stereo track where:
        - Left channel: Original Japanese audio (or background + reduced vocals)
        - Right channel: Chinese TTS audio aligned to Japanese timing
        """
        base = self._get_base_audio(mix_input)

        # Create a canvas for the Chinese track (same length as base)
        chinese_track = AudioSegment.silent(duration=len(base))

        for seg in mix_input.segments:
            tts_audio = self._load_tts_audio(seg["id"], mix_input.tts_results)
            if tts_audio is None:
                continue

            # Position Chinese audio at the start time of the Japanese segment
            position_ms = int(seg["start"] * 1000)

            # If TTS is longer than the segment, trim to fit
            segment_duration_ms = int((seg["end"] - seg["start"]) * 1000)
            if len(tts_audio) > segment_duration_ms:
                tts_audio = tts_audio[:segment_duration_ms]

            chinese_track = chinese_track.overlay(tts_audio, position=position_ms)

        # Ensure both tracks are mono
        base_mono = base.set_channels(1)
        chinese_mono = chinese_track.set_channels(1)

        # Match lengths
        max_len = max(len(base_mono), len(chinese_mono))
        base_mono = base_mono + AudioSegment.silent(duration=max_len - len(base_mono))
        chinese_mono = chinese_mono + AudioSegment.silent(duration=max_len - len(chinese_mono))

        # Create stereo: left = Japanese, right = Chinese
        result = AudioSegment.from_mono_audiosegments(base_mono, chinese_mono)
        logger.info("Dual mix complete: %dms", len(result))
        return result

    def _mix_overlay(self, mix_input: MixInput) -> AudioSegment:
        """Overlay mode: Chinese voice overlaid at lower volume on original.

        When vocals are separated, the background plays at full volume and
        original vocals are reduced, making room for the Chinese TTS.
        """
        base = self._get_base_audio(mix_input)
        result = base

        volume_db = self.settings.overlay_volume_db

        for seg in mix_input.segments:
            tts_audio = self._load_tts_audio(seg["id"], mix_input.tts_results)
            if tts_audio is None:
                continue

            # Position at segment start time
            position_ms = int(seg["start"] * 1000)

            # Adjust volume
            tts_audio = tts_audio + volume_db

            # If TTS is longer than the segment, trim it
            segment_duration_ms = int((seg["end"] - seg["start"]) * 1000)
            if len(tts_audio) > segment_duration_ms:
                tts_audio = tts_audio[:segment_duration_ms]

            # Overlay with ducking: reduce original volume during Chinese speech
            result = result.overlay(tts_audio, position=position_ms, gain_during_overlay=-3)

        logger.info("Overlay mix complete: %dms", len(result))
        return result

    def _mix_sequential(self, mix_input: MixInput) -> AudioSegment:
        """Sequential mode: Japanese segment followed by Chinese translation.

        Each Japanese segment is followed by its Chinese translation with a gap.
        This extends the total audio duration.

        When vocals are separated, background plays throughout and only
        vocals are used for the Japanese segments.
        """
        if mix_input.separated is not None:
            # Use background throughout, with vocals only during JP segments
            return self._mix_sequential_separated(mix_input)

        original = mix_input.original_audio
        gap = AudioSegment.silent(duration=self.settings.sequential_gap_ms)

        # Build a new audio track segment by segment
        result = AudioSegment.empty()
        last_end_ms = 0

        for seg in mix_input.segments:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)

            # Add any non-speech audio between segments
            if start_ms > last_end_ms:
                between = original[last_end_ms:start_ms]
                result += between

            # Add the Japanese segment
            jp_segment = original[start_ms:end_ms]
            result += jp_segment

            # Add gap
            result += gap

            # Add Chinese TTS
            tts_audio = self._load_tts_audio(seg["id"], mix_input.tts_results)
            if tts_audio is not None:
                result += tts_audio

            last_end_ms = end_ms

        # Add any remaining audio after the last segment
        if last_end_ms < len(original):
            result += original[last_end_ms:]

        logger.info("Sequential mix complete: %dms", len(result))
        return result

    def _mix_sequential_separated(self, mix_input: MixInput) -> AudioSegment:
        """Sequential mode with separated vocals/background.

        Background plays throughout. During JP segments, reduced vocals play.
        After each JP segment, Chinese TTS plays.
        """
        separated = mix_input.separated
        background = separated.background + self.settings.background_volume_db
        vocals = separated.vocals + self.settings.vocals_volume_db
        gap = AudioSegment.silent(duration=self.settings.sequential_gap_ms)

        # Start with background for the full duration
        result = AudioSegment.silent(duration=len(separated.original))

        # Overlay reduced vocals at full duration
        result = result.overlay(vocals)

        # Now overlay Chinese TTS after each JP segment
        for seg in mix_input.segments:
            tts_audio = self._load_tts_audio(seg["id"], mix_input.tts_results)
            if tts_audio is None:
                continue

            end_ms = int(seg["end"] * 1000)
            position_ms = end_ms + self.settings.sequential_gap_ms

            # Extend result if needed
            if position_ms + len(tts_audio) > len(result):
                result = result + AudioSegment.silent(
                    duration=position_ms + len(tts_audio) - len(result)
                )

            result = result.overlay(tts_audio, position=position_ms)

        logger.info("Sequential mix (separated) complete: %dms", len(result))
        return result

    def _mix_whisper(self, mix_input: MixInput) -> AudioSegment:
        """Whisper mode: Chinese voice at very low volume, like a subtle translation.

        When vocals are separated, background plays at full volume with reduced
        vocals, and Chinese TTS is barely audible underneath.
        """
        base = self._get_base_audio(mix_input)
        result = base

        volume_db = self.settings.whisper_volume_db

        for seg in mix_input.segments:
            tts_audio = self._load_tts_audio(seg["id"], mix_input.tts_results)
            if tts_audio is None:
                continue

            # Position at segment start time
            position_ms = int(seg["start"] * 1000)

            # Apply whisper volume
            tts_audio = tts_audio + volume_db

            # Trim if longer than segment
            segment_duration_ms = int((seg["end"] - seg["start"]) * 1000)
            if len(tts_audio) > segment_duration_ms:
                tts_audio = tts_audio[:segment_duration_ms]

            # Gentle overlay without ducking (preserve original atmosphere)
            result = result.overlay(tts_audio, position=position_ms)

        logger.info("Whisper mix complete: %dms", len(result))
        return result

    @staticmethod
    def export(
        audio: AudioSegment,
        output_path: Path | str,
        format: str = "mp3",
        bitrate: str = "192k",
        sample_rate: int = 44100,
    ) -> Path:
        """Export mixed audio to file.

        Args:
            audio: AudioSegment to export.
            output_path: Output file path.
            format: Audio format (mp3, wav, m4a, etc.).
            bitrate: Bitrate for lossy formats.
            sample_rate: Sample rate in Hz.

        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize sample rate
        audio = audio.set_frame_rate(sample_rate)

        audio.export(
            str(output_path),
            format=format,
            bitrate=bitrate,
        )
        logger.info("Exported audio to %s (%s, %s)", output_path, format, bitrate)
        return output_path