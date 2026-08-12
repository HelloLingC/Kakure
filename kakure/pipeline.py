"""Pipeline module - orchestrates the end-to-end bilingual audio generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kakure.asr import Segment, create_asr_processor
from kakure.checkpoint import (
    CheckpointStore,
    run_asr,
    run_separation,
    run_translation,
    run_tts,
)
from kakure.config import Settings, get_settings
from kakure.mixer import AudioMixer, MixInput
from kakure.separator import SeparatedAudio, VocalSeparator
from kakure.srt import srt_path as srt_output_path
from kakure.srt import write_srt
from kakure.translator import TranslatedSegment, Translator
from kakure.tts import create_tts_processor

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""

    output_path: Path
    segments: list[dict] = field(default_factory=list)
    mix_mode: str = ""
    duration_seconds: float = 0.0
    vocals_separated: bool = False
    srt_ja_path: Path | None = None
    srt_zh_path: Path | None = None
    srt_bilingual_path: Path | None = None


class Pipeline:
    """End-to-end pipeline for bilingual ASMR audio generation.

    Pipeline steps:
    1. ASR: Transcribe Japanese audio with timestamps
    2. Translation: Translate Japanese text to Chinese
    3. TTS: Generate Chinese voice audio
    4. Vocal separation: (Optional) Split original audio into vocals and background
    5. Mixing: Combine original and Chinese audio
    6. Export: Save the final bilingual audio
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.asr = create_asr_processor(self.settings)
        self.translator = Translator(self.settings)
        self.tts = create_tts_processor(self.settings)
        self.mixer = AudioMixer(self.settings)

    def run(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
    ) -> PipelineResult:
        """Run the full bilingual audio generation pipeline.

        Args:
            input_path: Path to the input Japanese ASMR audio file.
            output_path: Path for the output bilingual audio file.
                         Defaults to input_path with '_bilingual' suffix.

        Returns:
            PipelineResult with output path and metadata.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        if output_path is None:
            ext = input_path.suffix or f".{self.settings.output_format}"
            output_path = input_path.parent / f"{input_path.stem}_bilingual{ext}"
        else:
            output_path = Path(output_path)

        srt_dir = (
            Path(self.settings.srt_output_dir)
            if self.settings.srt_output_dir
            else input_path.parent
        )
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_ja_path = srt_output_path(input_path.stem, "ja", srt_dir)
        srt_zh_path = srt_output_path(input_path.stem, "zh", srt_dir)
        srt_bilingual_path = srt_output_path(input_path.stem, "bilingual", srt_dir)

        console.print("\n[bold cyan]Kakure[/] - ASMR Bilingual Voice Overlay")
        console.print(f"[dim]Input:  {input_path}[/]")
        console.print(f"[dim]Output: {output_path}[/]")
        console.print(f"[dim]Mode:    {self.settings.mix_mode.value}[/]")
        if self.settings.separate_vocals:
            console.print(
                f"[dim]Vocal separation: enabled (Demucs {self.settings.demucs_model.value})[/]"
            )
        console.print()

        separated: SeparatedAudio | None = None
        ckpt = CheckpointStore(input_path, self.settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Step 1: ASR - Transcribe Japanese audio
            task = progress.add_task("[cyan]Transcribing Japanese audio...", total=None)
            transcription, asr_cached = run_asr(ckpt, lambda: self.asr.transcribe(input_path))
            progress.update(task, completed=1, total=1)

            if not transcription.segments:
                raise ValueError("No speech segments detected in the audio file")

            if asr_cached:
                console.print(f"  [dim]✓ Found {len(transcription.segments)} segments (cached)[/]")
            else:
                console.print(
                    f"  [green]✓[/] Found {len(transcription.segments)} segments "
                    f"({transcription.duration:.1f}s, {transcription.language})"
                )
            write_srt(transcription.segments, srt_ja_path)

            # Step 2: Translation - Japanese to Chinese
            task = progress.add_task("[cyan]Translating to Chinese...", total=None)
            translated_segments, tr_cached = run_translation(
                ckpt,
                transcription.segments,
                lambda segs: self._translate_segments(segs),
            )
            progress.update(task, completed=1, total=1)

            if tr_cached:
                console.print(
                    f"  [dim]✓ Translated {len(translated_segments)} segments (cached)[/]"
                )
            else:
                console.print(f"  [green]✓[/] Translated {len(translated_segments)} segments")
            write_srt(translated_segments, srt_zh_path, text=lambda seg: seg.translated_text)

            # Step 3: TTS - Generate Chinese voice (per-segment resume)
            task = progress.add_task("[cyan]Generating Chinese voice...", total=None)
            segment_dicts = [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "original_text": seg.original_text,
                    "translated_text": seg.translated_text,
                }
                for seg in translated_segments
            ]
            tts_dicts, tts_reused = run_tts(
                ckpt,
                segment_dicts,
                lambda needed, outdir: [
                    {
                        "segment_id": r.segment_id,
                        "audio_path": str(r.audio_path),
                        "duration_ms": r.duration_ms,
                    }
                    for r in self.tts.generate_sync(needed, output_dir=outdir)
                ],
            )
            progress.update(task, completed=1, total=1)

            if tts_reused:
                console.print(
                    f"  [dim]✓ Generated {len(tts_dicts)} voice segments ({tts_reused} cached)[/]"
                )
            else:
                console.print(f"  [green]✓[/] Generated {len(tts_dicts)} voice segments")

            # Step 4: Vocal separation (optional)
            if self.settings.separate_vocals:
                task = progress.add_task("[cyan]Separating vocals from background...", total=None)
                separator = VocalSeparator(self.settings)
                separated, sep_cached = run_separation(
                    ckpt,
                    input_path,
                    lambda outdir: separator.separate(input_path, output_dir=outdir),
                )
                progress.update(task, completed=1, total=1)
                if sep_cached:
                    console.print("  [dim]✓ Vocals separated from background (cached)[/]")
                else:
                    console.print("  [green]✓[/] Vocals separated from background")
            else:
                separated = None

            # Step 5: Mixing - Combine audio tracks
            task = progress.add_task("[cyan]Mixing audio tracks...", total=None)
            from pydub import AudioSegment

            original_audio = AudioSegment.from_file(str(input_path))
            mix_input = MixInput(
                original_audio=original_audio,
                segments=segment_dicts,
                tts_results=tts_dicts,
                separated=separated,
            )
            mixed_audio = self.mixer.mix(mix_input)
            progress.update(task, completed=1, total=1)

            # Step 6: Export
            task = progress.add_task("[cyan]Exporting final audio...", total=None)
            output_format = output_path.suffix.lstrip(".") or self.settings.output_format
            if output_format == "mp3":
                bitrate = self.settings.output_bitrate
            else:
                bitrate = None

            self.mixer.export(
                mixed_audio,
                output_path,
                format=output_format,
                bitrate=bitrate or "192k",
                sample_rate=self.settings.output_sample_rate,
            )
            progress.update(task, completed=1, total=1)

        write_srt(
            translated_segments,
            srt_bilingual_path,
            text=lambda seg: f"{seg.original_text}\n{seg.translated_text}",
        )

        duration = len(mixed_audio) / 1000.0
        console.print(
            f"\n[bold green]✓ Done![/] Bilingual audio saved to: [bold]{output_path}[/]"
        )
        details = f"  Duration: {duration:.1f}s | Mode: {self.settings.mix_mode.value}"
        if separated:
            details += " | Vocals separated"
        console.print(details)

        return PipelineResult(
            output_path=output_path,
            segments=segment_dicts,
            mix_mode=self.settings.mix_mode.value,
            duration_seconds=duration,
            vocals_separated=separated is not None,
            srt_ja_path=srt_ja_path,
            srt_zh_path=srt_zh_path,
            srt_bilingual_path=srt_bilingual_path,
        )

    def _translate_segments(self, segments: list[Segment]) -> list[TranslatedSegment]:
        """Convert ASR segments to TranslatedSegments and translate them."""
        translated = [
            TranslatedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text="",  # Will be filled by translator
            )
            for seg in segments
        ]
        return self.translator.translate_segments(translated)
