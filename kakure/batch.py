"""Batch module - process multiple audio files headlessly.

Runs the standard pipeline (ASR -> translation -> TTS -> separation -> mix ->
export) over a list of audio files sequentially, reusing a single
:class:`~kakure.pipeline.Pipeline` so models are loaded once instead of once
per file. Per-file checkpoints (keyed by input SHA-256) make re-runs resume
where they left off.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kakure.config import Settings, get_settings
from kakure.pipeline import Pipeline

logger = logging.getLogger(__name__)
console = Console()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".aac", ".wma", ".opus"}


@dataclass
class BatchResult:
    """Summary of a batch run."""

    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _has_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def collect_audio_files(inputs: list[str], recursive: bool = False) -> list[Path]:
    """Expand directories, files and glob patterns into a sorted, deduped list.

    Directories yield their audio files directly (``*``) or recursively
    (``**/*``). Explicit file paths are kept as-is. Anything else is treated
    as a glob pattern.
    """
    found: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            pattern = str(p / ("**/*" if recursive else "*"))
            candidates = [Path(x) for x in _glob.glob(pattern, recursive=True)]
        elif _has_magic(raw):
            candidates = [Path(x) for x in _glob.glob(raw, recursive=recursive)]
        else:
            candidates = [p]
        found.extend(c for c in candidates if _is_audio(c))

    seen: set[str] = set()
    unique: list[Path] = []
    for f in found:
        key = os.path.normcase(str(f))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _output_path_for(input_path: Path, output_dir: Path | None, output_format: str) -> Path:
    ext = f".{output_format}"
    if output_dir:
        return output_dir / f"{input_path.stem}_bilingual{ext}"
    return input_path.parent / f"{input_path.stem}_bilingual{ext}"


def run_batch(
    inputs: list[str],
    settings: Settings | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> BatchResult:
    """Process every matched audio file through the full pipeline.

    Args:
        inputs: Directories, files or glob patterns to process.
        settings: Application settings. Loaded from ``kakure.toml`` if None.
        output_dir: Directory for outputs. Defaults to next to each input.
        recursive: Recurse into subdirectories of directory inputs.
        force: Reprocess files whose output already exists.
        dry_run: List files that would be processed and exit.

    Returns:
        BatchResult with per-file outcomes.
    """
    settings = settings or get_settings()
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(inputs, recursive=recursive)
    result = BatchResult(total=len(files))

    if not files:
        console.print("[yellow]No audio files found.[/]")
        return result

    output_dir_path = Path(output_dir) if output_dir else None
    if output_dir_path is None and settings.output_dir:
        output_dir_path = Path(settings.output_dir)
    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Kakure batch[/] - {len(files)} file(s)")
    console.print(f"[dim]Mode: {settings.mix_mode.value}[/]")
    if output_dir_path:
        console.print(f"[dim]Output: {output_dir_path}[/]")
    console.print()

    if dry_run:
        for f in files:
            out = _output_path_for(f, output_dir_path, settings.output_format)
            status = "[yellow](exists)[/]" if out.exists() else ""
            console.print(f"  [dim]{f}[/] -> {out} {status}")
        return result

    pipeline = Pipeline(settings)
    start = time.perf_counter()

    for idx, f in enumerate(files, 1):
        out = _output_path_for(f, output_dir_path, settings.output_format)
        console.print(f"\n[{idx}/{len(files)}] [bold]{f.name}[/]")

        if out.exists() and not force:
            console.print(f"  [dim]Skip: output exists ({out.name})[/]")
            result.skipped += 1
            continue

        try:
            pipeline.run(f, output_path=out)
            result.succeeded += 1
            result.outputs.append(str(out))
        except Exception as e:
            result.failed += 1
            result.failures.append((str(f), str(e)))
            logger.exception("Batch file failed: %s", f)
            console.print(f"  [red]Failed: {e}[/]")

    _print_summary(result, time.perf_counter() - start)
    return result


def _print_summary(result: BatchResult, elapsed: float) -> None:
    console.print("\n[bold]Batch complete[/]")
    table = Table(show_header=False, box=None)
    table.add_row("Files", str(result.total))
    table.add_row("Succeeded", f"[green]{result.succeeded}[/]")
    table.add_row("Skipped", f"[yellow]{result.skipped}[/]")
    table.add_row("Failed", f"[red]{result.failed}[/]")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    console.print(table)

    if result.failures:
        console.print("\n[bold red]Failures:[/]")
        for path, err in result.failures:
            console.print(f"  [red]x[/] {path}: {err}")
