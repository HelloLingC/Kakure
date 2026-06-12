"""CLI module - Command-line interface for Kakure."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kakure.config import (
    ASRBackend,
    ChineseVoice,
    DemucsModel,
    KotobaWhisperModel,
    MixMode,
    Settings,
    TTSBackend,
    WhisperModelSize,
    get_settings,
)

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="kakure")
def cli():
    """Kakure - ASMR Japanese-to-Chinese bilingual voice overlay tool.

    Translates Japanese ASMR voice audio into bilingual voice audio
    by overlaying a Chinese voice track into the original audio.
    """
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", "output_path", type=click.Path(path_type=Path), default=None,
    help="Output file path. Defaults to input_path with '_bilingual' suffix.",
)
@click.option(
    "-m", "--mode", type=click.Choice([m.value for m in MixMode]), default=None,
    help="Mixing mode: dual, overlay, sequential, whisper.",
)
@click.option(
    "--asr-backend", type=click.Choice([b.value for b in ASRBackend]), default=None,
    help="ASR backend: faster-whisper or kotoba-whisper.",
)
@click.option(
    "--model", "whisper_model", type=click.Choice([s.value for s in WhisperModelSize]),
    default=None, help="Whisper model size (faster-whisper backend).",
)
@click.option(
    "--kotoba-model", type=click.Choice([m.value for m in KotobaWhisperModel]),
    default=None, help="Kotoba-whisper model (kotoba-whisper backend).",
)
@click.option(
    "--tts-backend", type=click.Choice([b.value for b in TTSBackend]), default=None,
    help="TTS backend: edge-tts or indextts.",
)
@click.option(
    "--voice", type=click.Choice([v.value for v in ChineseVoice]), default=None,
    help="Chinese TTS voice (edge-tts backend).",
)
@click.option(
    "--reference-audio", type=click.Path(exists=True, path_type=Path), default=None,
    help="Reference audio for voice cloning (required for IndexTTS backend).",
)
@click.option(
    "--volume-overlay", type=float, default=None,
    help="Chinese voice volume in overlay mode (dB, negative = quieter).",
)
@click.option(
    "--volume-whisper", type=float, default=None,
    help="Chinese voice volume in whisper mode (dB, negative = quieter).",
)
@click.option(
    "--format", "output_format", type=click.Choice(["mp3", "wav", "m4a", "flac", "ogg"]),
    default=None, help="Output audio format.",
)
@click.option(
    "--bitrate", default=None, help="Output bitrate (e.g., '192k', '320k').",
)
@click.option(
    "--device", type=click.Choice(["cpu", "cuda"]), default=None,
    help="Whisper inference device.",
)
@click.option(
    "--separate-vocals", is_flag=True, default=False,
    help="Use Demucs to separate vocals from background before mixing.",
)
@click.option(
    "--demucs-model", type=click.Choice([m.value for m in DemucsModel]), default=None,
    help="Demucs model for vocal separation.",
)
@click.option(
    "--vocals-volume", type=float, default=None,
    help="Original vocals volume when separated (dB, negative = quieter).",
)
@click.option(
    "--background-volume", type=float, default=None,
    help="Background volume when separated (dB).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def process(
    input_path: Path,
    output_path: Path | None,
    mode: str | None,
    asr_backend: str | None,
    whisper_model: str | None,
    kotoba_model: str | None,
    tts_backend: str | None,
    voice: str | None,
    reference_audio: Path | None,
    volume_overlay: float | None,
    volume_whisper: float | None,
    output_format: str | None,
    bitrate: str | None,
    device: str | None,
    separate_vocals: bool,
    demucs_model: str | None,
    vocals_volume: float | None,
    background_volume: float | None,
    verbose: bool,
):
    """Process an ASMR audio file to create bilingual output.

    INPUT_PATH is the path to the Japanese ASMR audio file.
    """
    setup_logging(verbose)

    # Load settings and apply CLI overrides
    settings = get_settings()
    if mode:
        settings.mix_mode = MixMode(mode)
    if asr_backend:
        settings.asr_backend = ASRBackend(asr_backend)
    if whisper_model:
        settings.whisper_model = WhisperModelSize(whisper_model)
    if kotoba_model:
        settings.kotoba_whisper_model = KotobaWhisperModel(kotoba_model)
    if tts_backend:
        settings.tts_backend = TTSBackend(tts_backend)
    if voice:
        settings.chinese_voice = ChineseVoice(voice)
    if reference_audio:
        settings.indextts_reference_audio = reference_audio
    if volume_overlay is not None:
        settings.overlay_volume_db = volume_overlay
    if volume_whisper is not None:
        settings.whisper_volume_db = volume_whisper
    if output_format:
        settings.output_format = output_format
    if bitrate:
        settings.output_bitrate = bitrate
    if device:
        settings.whisper_device = device
    if separate_vocals:
        settings.separate_vocals = True
    if demucs_model:
        settings.demucs_model = DemucsModel(demucs_model)
    if vocals_volume is not None:
        settings.vocals_volume_db = vocals_volume
    if background_volume is not None:
        settings.background_volume_db = background_volume

    from kakure.pipeline import Pipeline

    pipeline = Pipeline(settings)
    try:
        result = pipeline.run(input_path, output_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--asr-backend", type=click.Choice([b.value for b in ASRBackend]), default=None,
    help="ASR backend: faster-whisper or kotoba-whisper.",
)
@click.option(
    "--model", "whisper_model", type=click.Choice([s.value for s in WhisperModelSize]),
    default=None, help="Whisper model size (faster-whisper backend).",
)
@click.option(
    "--kotoba-model", type=click.Choice([m.value for m in KotobaWhisperModel]),
    default=None, help="Kotoba-whisper model (kotoba-whisper backend).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def transcribe(
    input_path: Path,
    asr_backend: str | None,
    whisper_model: str | None,
    kotoba_model: str | None,
    verbose: bool,
):
    """Transcribe a Japanese audio file (ASR only, no translation).

    INPUT_PATH is the path to the Japanese audio file.
    """
    setup_logging(verbose)

    settings = get_settings()
    if asr_backend:
        settings.asr_backend = ASRBackend(asr_backend)
    if whisper_model:
        settings.whisper_model = WhisperModelSize(whisper_model)
    if kotoba_model:
        settings.kotoba_whisper_model = KotobaWhisperModel(kotoba_model)

    from kakure.asr import create_asr_processor

    asr = create_asr_processor(settings)
    result = asr.transcribe(input_path)

    table = Table(title=f"Transcription: {input_path.name}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Start", style="cyan", width=8)
    table.add_column("End", style="cyan", width=8)
    table.add_column("Text", style="white")

    for seg in result.segments:
        table.add_row(
            str(seg.id),
            f"{seg.start:.2f}",
            f"{seg.end:.2f}",
            seg.text,
        )

    console.print(table)
    console.print(
        f"\n[green]Language: {result.language} "
        f"({result.language_probability:.1%} confidence)[/]"
    )
    console.print(f"[green]Duration: {result.duration:.1f}s[/]")


@cli.command()
def voices():
    """List available Chinese TTS voices (edge-tts)."""
    from kakure.tts import EdgeTTSProcessor

    console.print("[bold]Fetching available Chinese voices (edge-tts)...[/]\n")
    voice_list = EdgeTTSProcessor.list_chinese_voices_sync()

    table = Table(title="Available Chinese TTS Voices (edge-tts)")
    table.add_column("Voice ID", style="cyan")
    table.add_column("Gender", style="green")
    table.add_column("Locale", style="dim")

    for v in voice_list:
        table.add_row(v["ShortName"], v["Gender"], v["Locale"])

    console.print(table)
    console.print("\n[dim]IndexTTS uses voice cloning from a reference audio file, not pre-built voices.[/]")


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Kakure Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    config_items = [
        ("ASR Backend", settings.asr_backend.value),
        ("Whisper Model", settings.whisper_model.value),
        ("Kotoba-Whisper Model", settings.kotoba_whisper_model.value),
        ("Whisper Device", settings.whisper_device),
        ("Whisper Compute Type", settings.whisper_compute_type),
        ("TTS Backend", settings.tts_backend.value),
        ("Chinese Voice (edge-tts)", settings.chinese_voice.value),
        ("IndexTTS Reference Audio", str(settings.indextts_reference_audio or "Not set")),
        ("IndexTTS Model Dir", str(settings.indextts_model_dir or "Auto-download")),
        ("Translation Backend", settings.translation_backend.value),
        ("Separate Vocals", str(settings.separate_vocals)),
        ("Demucs Model", settings.demucs_model.value),
        ("Demucs Device", settings.demucs_device),
        ("Vocals Volume (dB)", str(settings.vocals_volume_db)),
        ("Background Volume (dB)", str(settings.background_volume_db)),
        ("Mix Mode", settings.mix_mode.value),
        ("Overlay Volume (dB)", str(settings.overlay_volume_db)),
        ("Whisper Volume (dB)", str(settings.whisper_volume_db)),
        ("Sequential Gap (ms)", str(settings.sequential_gap_ms)),
        ("Output Format", settings.output_format),
        ("Output Bitrate", settings.output_bitrate),
        ("Output Sample Rate", str(settings.output_sample_rate)),
        ("Temp Dir", str(settings.temp_dir)),
    ]

    for key, val in config_items:
        table.add_row(key, val)

    console.print(table)


if __name__ == "__main__":
    cli()