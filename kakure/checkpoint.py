"""Checkpoint module - cache pipeline intermediate results for resumability.

The pipeline stages (ASR, translation, TTS, vocal separation) are the slow,
expensive parts of a run. Their results are cached on disk keyed by the SHA-256
of the input audio file, under ``<checkpoint_dir>/checkpoints/<hash>/`` by
default (``checkpoint_dir`` empty -> ``<temp_dir>/checkpoints``).

Each stage stores a JSON record with:

- ``input_hash``: SHA-256 of the source audio file
- ``settings_fp``: hash of the subset of :class:`Settings` fields the stage
  depends on (changing, say, the TTS voice invalidates the TTS cache)
- ``upstream_fp``: hash of the upstream artifact the stage is derived from
  (the input file for ASR/separation, the ASR record for translation, the
  translation record for TTS). Recomputing a stage therefore automatically
  invalidates everything downstream.

TTS resumes per-segment: a segment's audio is reused only when its stored
text-hash matches the current translated text and the file still exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from kakure.asr import Segment, TranscriptionResult, Word
from kakure.config import Settings
from kakure.translator import TranslatedSegment

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1

# Subset of Settings fields each stage depends on. Changing any of these
# invalidates that stage's checkpoint (and, via upstream hashes, all stages
# built on top of it).
STAGE_SETTING_KEYS: dict[str, list[str]] = {
    "asr": [
        "asr_backend",
        "whisper_model",
        "whisper_device",
        "whisper_compute_type",
        "whisper_language",
        "whisper_beam_size",
        "whisper_vad_filter",
        "model_dir",
        "kotoba_whisper_model",
        "kotoba_whisper_chunk_length_s",
    ],
    "translate": [
        "translation_backend",
        "openai_model",
        "openai_base_url",
        "translation_prompt",
        "translation_batch_size",
        "translation_batch_token_limit",
        "translation_max_concurrency",
    ],
    "tts": [
        "tts_backend",
        "audiocpp_family",
        "audiocpp_model",
        "audiocpp_language",
        "audiocpp_reference_audio",
        "audiocpp_reference_text",
        "audiocpp_speed",
        "audiocpp_max_tokens",
        "audiocpp_backend",
        "audiocpp_device",
        "model_dir",
    ],
    "separate": [
        "demucs_model",
        "demucs_device",
        "model_dir",
    ],
}


def _str_sha256(value: str) -> str:
    """SHA-256 of a string."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _file_sha256(path: Path | str) -> str:
    """Streaming SHA-256 of a file. Returns '' if the file doesn't exist."""
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_root(settings: Settings) -> Path:
    """Return the root directory that holds per-input checkpoint folders."""
    if settings.checkpoint_dir:
        return Path(settings.checkpoint_dir)
    return Path(settings.temp_dir) / "checkpoints"


def clear_all_checkpoints(settings: Settings) -> int:
    """Delete all cached checkpoints and return the number of entries cleared."""
    root = checkpoint_root(settings)
    if not root.is_dir():
        return 0
    entries = [p for p in root.iterdir() if p.is_dir()]
    shutil.rmtree(root, ignore_errors=True)
    return len(entries)


# ---------------------------------------------------------------------------
# ASR serialization
# ---------------------------------------------------------------------------


def segment_to_dict(segment: Segment) -> dict:
    """Serialize an ASR Segment to a JSON-safe dict."""
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [
            {
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "probability": w.probability,
            }
            for w in segment.words
        ],
    }


def segment_from_dict(data: dict) -> Segment:
    """Deserialize an ASR Segment from a dict (see :func:`segment_to_dict`)."""
    return Segment(
        id=data["id"],
        start=data["start"],
        end=data["end"],
        text=data["text"],
        words=[
            Word(
                start=w["start"],
                end=w["end"],
                word=w["word"],
                probability=w["probability"],
            )
            for w in data.get("words", [])
        ],
    )


def transcription_to_dict(result: TranscriptionResult) -> dict:
    """Serialize a TranscriptionResult to a JSON-safe dict."""
    return {
        "segments": [segment_to_dict(s) for s in result.segments],
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
    }


def transcription_from_dict(data: dict) -> TranscriptionResult:
    """Deserialize a TranscriptionResult from a dict."""
    return TranscriptionResult(
        segments=[segment_from_dict(s) for s in data["segments"]],
        language=data.get("language", "ja"),
        language_probability=data.get("language_probability", 1.0),
        duration=data.get("duration", 0.0),
    )


def translated_segments_to_dict(segments: list[TranslatedSegment]) -> dict:
    """Serialize translated segments to a JSON-safe dict."""
    return {
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "original_text": s.original_text,
                "translated_text": s.translated_text,
            }
            for s in segments
        ]
    }


def translated_segments_from_dict(data: dict) -> list[TranslatedSegment]:
    """Deserialize translated segments from a dict."""
    return [TranslatedSegment(**s) for s in data["segments"]]


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Persists and validates pipeline stage results for a single input file."""

    def __init__(self, input_path: Path | str, settings: Settings):
        self.settings = settings
        self.enabled = bool(getattr(settings, "enable_checkpoints", True))
        self.input_path = Path(input_path)
        self.input_hash = _file_sha256(self.input_path)
        self.root = checkpoint_root(settings)
        self.dir = self.root / self.input_hash

    # -- paths -------------------------------------------------------------

    def stage_meta_path(self, stage: str) -> Path:
        """Path of the JSON record for a stage."""
        return self.dir / f"{stage}.json"

    @property
    def tts_dir(self) -> Path:
        """Directory holding per-segment TTS audio files."""
        return self.dir / "tts"

    @property
    def separated_dir(self) -> Path:
        """Directory holding separated vocals/background WAV files."""
        return self.dir / "separated"

    # -- read/write ----------------------------------------------------------

    def load(self, stage: str) -> dict | None:
        """Return the valid checkpoint record for ``stage``, else None.

        A record is only valid when its version, settings fingerprint and
        upstream fingerprint all match the current input/settings, and its
        external artifacts (e.g. separated WAVs) still exist.
        """
        if not self.enabled or not self.input_hash:
            return None
        path = self.stage_meta_path(stage)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Ignoring corrupt checkpoint %s", path)
            return None
        if (
            record.get("version") != CHECKPOINT_VERSION
            or record.get("stage") != stage
            or record.get("settings_fp") != self._settings_fp(stage)
            or record.get("upstream_fp") != self._upstream_fp(stage)
            or not self._artifacts_present(stage, record.get("data") or {})
        ):
            return None
        return record

    def save(self, stage: str, data: dict) -> None:
        """Atomically persist a stage checkpoint."""
        if not self.enabled or not self.input_hash:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {
            "version": CHECKPOINT_VERSION,
            "stage": stage,
            "created": datetime.now(timezone.utc).isoformat(),
            "input_hash": self.input_hash,
            "settings_fp": self._settings_fp(stage),
            "upstream_fp": self._upstream_fp(stage),
            "data": data,
        }
        target = self.stage_meta_path(stage)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)

    def purge_stage_artifacts(self, stage: str) -> None:
        """Delete the on-disk artifacts of a stage (used before regeneration)."""
        if not self.enabled:
            return
        if stage == "tts":
            shutil.rmtree(self.tts_dir, ignore_errors=True)
        elif stage == "separate":
            shutil.rmtree(self.separated_dir, ignore_errors=True)

    # -- validation internals --------------------------------------------------

    def _settings_value(self, key: str):
        value = getattr(self.settings, key, None)
        return value.value if isinstance(value, Enum) else value

    def _settings_fp(self, stage: str) -> str:
        keys = STAGE_SETTING_KEYS.get(stage, [])
        values = {k: self._settings_value(k) for k in keys}
        return _str_sha256(json.dumps(values, sort_keys=True, ensure_ascii=False))

    def _upstream_fp(self, stage: str) -> str:
        if stage in ("asr", "separate"):
            return self.input_hash
        if stage == "translate":
            return _file_sha256(self.stage_meta_path("asr"))
        if stage == "tts":
            return _file_sha256(self.stage_meta_path("translate"))
        raise ValueError(f"Unknown checkpoint stage: {stage}")

    def _artifacts_present(self, stage: str, data: dict) -> bool:
        if stage == "separate":
            return (self.separated_dir / "vocals.wav").is_file() and (
                self.separated_dir / "background.wav"
            ).is_file()
        return True


# ---------------------------------------------------------------------------
# Stage builders (shared by the CLI pipeline and the web API)
# ---------------------------------------------------------------------------


def run_asr(store: CheckpointStore, transcribe_fn) -> tuple[TranscriptionResult, bool]:
    """Run ASR unless a valid checkpoint exists.

    Returns ``(result, from_cache)``.
    """
    record = store.load("asr")
    if record is not None:
        logger.info("Checkpoint hit for ASR stage")
        return transcription_from_dict(record["data"]), True
    result = transcribe_fn()
    store.save("asr", transcription_to_dict(result))
    return result, False


def run_translation(
    store: CheckpointStore,
    segments: list[Segment],
    translate_fn,
) -> tuple[list[TranslatedSegment], bool]:
    """Run translation unless a valid checkpoint exists.

    ``translate_fn`` receives the ASR segments and returns translated segments.
    Returns ``(segments, from_cache)``.
    """
    record = store.load("translate")
    if record is not None:
        logger.info("Checkpoint hit for translation stage")
        return translated_segments_from_dict(record["data"]), True
    result = translate_fn(segments)
    store.save("translate", translated_segments_to_dict(result))
    return result, False


def run_tts(
    store: CheckpointStore,
    segment_dicts: list[dict],
    generate_fn,
) -> tuple[list[dict], int]:
    """Generate TTS audio with per-segment resume.

    ``generate_fn`` receives ``(needed_segments, output_dir)`` and returns the
    list of result dicts ``{segment_id, audio_path, duration_ms}`` for the
    segments it synthesized. A cached segment is reused when its stored
    text-hash matches the current translated text and its audio file still
    exists; otherwise it is regenerated. An invalid manifest (e.g. settings or
    upstream text changed) purges the whole TTS directory first.

    Returns ``(all_result_dicts, reused_count)``.
    """
    text_by_id = {seg["id"]: seg["translated_text"] for seg in segment_dicts}

    record = store.load("tts")
    cached: dict[int, dict] = {}
    if record is not None:
        data = record["data"]
        stored_hashes = data.get("text_hashes", {})
        for entry in data.get("results", []):
            sid = int(entry["segment_id"])
            if sid in text_by_id and stored_hashes.get(str(sid)) == _str_sha256(
                text_by_id[sid]
            ):
                if Path(entry["audio_path"]).is_file():
                    cached[sid] = entry
    else:
        # Stale config or upstream data — start the TTS stage fresh.
        store.purge_stage_artifacts("tts")

    reused = len(cached)
    needed = [seg for seg in segment_dicts if seg["id"] not in cached]
    if needed:
        fresh = generate_fn(needed, store.tts_dir)
        for entry in fresh:
            cached[int(entry["segment_id"])] = entry

    final: list[dict] = []
    hashes: dict[str, str] = {}
    for seg in segment_dicts:
        entry = cached.get(seg["id"])
        if entry is None:
            continue
        final.append(entry)
        hashes[str(seg["id"])] = _str_sha256(text_by_id[seg["id"]])
    store.save("tts", {"results": final, "text_hashes": hashes})
    return final, reused


def run_separation(
    store: CheckpointStore,
    input_path: Path | str,
    separate_fn,
):
    """Run vocal separation unless valid separated WAVs are cached.

    ``separate_fn`` receives the output directory and returns a
    :class:`kakure.separator.SeparatedAudio`. Returns ``(audio, from_cache)``.
    """
    from kakure.separator import load_separated

    record = store.load("separate")
    if record is not None:
        separated = load_separated(store.separated_dir, input_path)
        if separated is not None:
            logger.info("Checkpoint hit for vocal separation stage")
            return separated, True
    store.purge_stage_artifacts("separate")
    store.separated_dir.mkdir(parents=True, exist_ok=True)
    separated = separate_fn(store.separated_dir)
    store.save("separate", {"duration_ms": len(separated.vocals)})
    return separated, False
