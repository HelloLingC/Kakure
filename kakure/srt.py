"""SRT subtitle export helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol


class _SegmentLike(Protocol):
    """Anything with start/end times in seconds (ASR or translated segments)."""

    start: float
    end: float


def srt_path(stem: str, kind: str, output_dir: Path | str | None = None) -> Path:
    """Return the path for an SRT file: ``{stem}_{kind}.srt``.

    Args:
        stem: Base name of the output (e.g. the input audio filename stem).
        kind: SRT variant, e.g. ``ja``, ``zh``, ``bilingual``.
        output_dir: Directory to write into. Defaults to the current directory.
    """
    base = Path(output_dir) if output_dir else Path.cwd()
    return base / f"{stem}_{kind}.srt"


def format_timestamp(seconds: float) -> str:
    """Format a time in seconds as an SRT timestamp (``HH:MM:SS,mmm``)."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(
    segments: Iterable[_SegmentLike],
    path: Path | str,
    text: Callable[..., str] = lambda seg: seg.text,
) -> Path:
    """Write segments to an SRT subtitle file (UTF-8).

    Args:
        segments: Segments with ``start``/``end`` attributes (seconds).
            Entries are renumbered sequentially from 1.
        path: Destination ``.srt`` file path.
        text: Callable extracting the subtitle text from a segment.
            Defaults to ``seg.text`` (ASR segments); pass
            ``lambda s: s.translated_text`` for translated segments.

    Returns:
        The path written to.
    """
    path = Path(path)
    blocks = [
        f"{idx}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{text(seg)}"
        for idx, seg in enumerate(segments, start=1)
    ]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path
