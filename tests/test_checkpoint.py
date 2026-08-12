"""Tests for the checkpoint mechanism (kakure.checkpoint)."""

from __future__ import annotations

from pathlib import Path

import kakure.separator as separator_module
from kakure.asr import Segment, TranscriptionResult, Word
from kakure.checkpoint import (
    CheckpointStore,
    clear_all_checkpoints,
    run_asr,
    run_separation,
    run_translation,
    run_tts,
    transcription_from_dict,
    transcription_to_dict,
    translated_segments_from_dict,
    translated_segments_to_dict,
)
from kakure.config import Settings
from kakure.translator import TranslatedSegment


def _make_store(tmp_path, **settings_kwargs) -> tuple[CheckpointStore, Path]:
    """Build a CheckpointStore with an input file under a temp dir."""
    settings = Settings(
        enable_checkpoints=True,
        temp_dir=str(tmp_path),
        checkpoint_dir="",
        **settings_kwargs,
    )
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake-audio-content")
    return CheckpointStore(input_path, settings), input_path


def _result_a() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.5,
                text="おはよう",
                words=[Word(0.0, 0.5, "お", 0.9)],
            )
        ],
        language="ja",
        language_probability=0.98,
        duration=1.5,
    )


def _result_b() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="おはよう"),
            Segment(id=1, start=1.5, end=3.0, text="こんにちは"),
        ],
        language="ja",
        language_probability=0.99,
        duration=3.0,
    )


def _translated() -> list[TranslatedSegment]:
    return [
        TranslatedSegment(
            id=0, start=0.0, end=1.5, original_text="おはよう", translated_text="早上好"
        ),
        TranslatedSegment(
            id=1, start=1.5, end=3.0, original_text="こんにちは", translated_text="你好"
        ),
    ]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_transcription_round_trip():
    data = transcription_to_dict(_result_a())
    result = transcription_from_dict(data)
    assert result.segments[0].text == "おはよう"
    assert result.segments[0].words[0].word == "お"
    assert result.language == "ja"
    assert result.duration == 1.5


def test_translated_segments_round_trip():
    data = translated_segments_to_dict(_translated())
    result = translated_segments_from_dict(data)
    assert [s.translated_text for s in result] == ["早上好", "你好"]
    assert result[1].id == 1


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))
    record = store.load("asr")
    assert record is not None
    assert record["stage"] == "asr"
    assert transcription_from_dict(record["data"]).segments[0].text == "おはよう"


def test_missing_stage_returns_none(tmp_path):
    store, _ = _make_store(tmp_path)
    assert store.load("asr") is None


def test_corrupt_json_invalidates(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))
    store.stage_meta_path("asr").write_text("{not json!!", encoding="utf-8")
    assert store.load("asr") is None


def test_settings_change_invalidates(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))

    other, _ = _make_store(tmp_path, whisper_model="tiny", whisper_language="ja")
    assert other.load("asr") is None


def test_input_change_uses_new_dir(tmp_path):
    store, input_path = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))
    input_path.write_bytes(b"different-content")
    other = CheckpointStore(input_path, store.settings)
    assert other.dir != store.dir
    assert other.load("asr") is None


def test_upstream_change_invalidates_downstream(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))
    store.save("translate", translated_segments_to_dict(_translated()))
    assert store.load("translate") is not None

    # Recompute ASR with different output -> translate checkpoint must be stale
    store.save("asr", transcription_to_dict(_result_b()))
    assert store.load("translate") is None


def test_clear_all_checkpoints(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))

    # Different input content -> a separate checkpoint directory
    other_input = tmp_path / "other.wav"
    other_input.write_bytes(b"different-content")
    other = CheckpointStore(other_input, store.settings)
    other.save("asr", transcription_to_dict(_result_b()))

    assert clear_all_checkpoints(store.settings) == 2
    assert not store.root.exists()
    assert clear_all_checkpoints(store.settings) == 0


def test_disabled_store_does_not_persist(tmp_path):
    settings = Settings(enable_checkpoints=False, temp_dir=str(tmp_path))
    store = CheckpointStore(tmp_path / "input.wav", settings)
    store.save("asr", transcription_to_dict(_result_a()))
    assert store.load("asr") is None
    assert not store.root.exists()


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------


def test_run_asr_cache_miss_then_hit(tmp_path):
    store, _ = _make_store(tmp_path)
    calls = []

    def transcribe():
        calls.append(1)
        return _result_a()

    result, cached = run_asr(store, transcribe)
    assert result.segments[0].text == "おはよう"
    assert cached is False
    assert len(calls) == 1

    result, cached = run_asr(store, transcribe)
    assert cached is True
    assert len(calls) == 1  # compute not called again


def test_run_translation_cache(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("asr", transcription_to_dict(_result_a()))
    calls = []

    def translate(segs):
        calls.append(len(segs))
        return _translated()

    result, cached = run_translation(store, _result_a().segments, translate)
    assert cached is False
    assert result[0].translated_text == "早上好"

    result, cached = run_translation(store, _result_a().segments, translate)
    assert cached is True
    assert len(calls) == 1


def _tts_segment_dicts():
    return [
        {"id": 0, "start": 0.0, "end": 1.0, "original_text": "あ", "translated_text": "啊"},
        {"id": 1, "start": 1.0, "end": 2.0, "original_text": "い", "translated_text": "咦"},
    ]


def _make_tts_generate(calls: list):
    def generate(needed, outdir):
        calls.append([seg["id"] for seg in needed])
        outdir.mkdir(parents=True, exist_ok=True)
        entries = []
        for seg in needed:
            path = outdir / f"segment_{seg['id']:04d}.mp3"
            path.write_bytes(b"fake-audio")
            entries.append({"segment_id": seg["id"], "audio_path": str(path), "duration_ms": 500})
        return entries

    return generate


def test_run_tts_full_cache(tmp_path):
    store, _ = _make_store(tmp_path)
    calls: list = []
    generate = _make_tts_generate(calls)

    results, reused = run_tts(store, _tts_segment_dicts(), generate)
    assert reused == 0
    assert len(results) == 2
    assert calls == [[0, 1]]

    results, reused = run_tts(store, _tts_segment_dicts(), generate)
    assert reused == 2
    assert calls == [[0, 1]]  # generate not called on second run


def test_run_tts_partial_resume(tmp_path):
    store, _ = _make_store(tmp_path)
    calls: list = []
    generate = _make_tts_generate(calls)

    run_tts(store, _tts_segment_dicts(), generate)

    # Change the translation of segment 1 -> only that one is regenerated
    segs = _tts_segment_dicts()
    segs[1]["translated_text"] = "咦？"
    calls.clear()
    results, reused = run_tts(store, segs, generate)
    assert reused == 1
    assert calls == [[1]]
    assert len(results) == 2


def test_run_tts_regenerates_missing_file(tmp_path):
    store, _ = _make_store(tmp_path)
    calls: list = []
    generate = _make_tts_generate(calls)

    results, _ = run_tts(store, _tts_segment_dicts(), generate)
    # Delete one audio file -> should be regenerated
    (store.tts_dir / "segment_0000.mp3").unlink()
    calls.clear()
    results, reused = run_tts(store, _tts_segment_dicts(), generate)
    assert reused == 1
    assert calls == [[0]]
    assert len(results) == 2


def test_run_separation_cache(tmp_path, monkeypatch):
    store, input_path = _make_store(tmp_path)
    calls: list = []

    class _FakeSeparated:
        vocals = [0]

    def separate_fn(outdir):
        calls.append(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "vocals.wav").write_bytes(b"v")
        (outdir / "background.wav").write_bytes(b"b")
        return _FakeSeparated()

    # Cache miss path
    monkeypatch.setattr(separator_module, "load_separated", lambda *a, **k: "separated-object")
    separated, cached = run_separation(store, input_path, separate_fn)
    assert cached is False
    assert len(calls) == 1

    # Cache hit path: store.load("separate") must validate wav files on disk
    monkeypatch.setattr(separator_module, "load_separated", lambda *a, **k: "cached-object")
    separated, cached = run_separation(store, input_path, separate_fn)
    assert cached is True
    assert separated == "cached-object"
    assert len(calls) == 1


def test_run_separation_missing_wav_recomputes(tmp_path, monkeypatch):
    store, input_path = _make_store(tmp_path)
    calls: list = []
    monkeypatch.setattr(separator_module, "load_separated", lambda *a, **k: None)
    store.separated_dir.mkdir(parents=True, exist_ok=True)
    (store.separated_dir / "vocals.wav").write_bytes(b"v")
    (store.separated_dir / "background.wav").write_bytes(b"b")

    def separate_fn(outdir):
        calls.append(1)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "vocals.wav").write_bytes(b"v2")
        (outdir / "background.wav").write_bytes(b"b2")
        return type("Fake", (), {"vocals": [0]})()

    # No record saved yet -> load("separate") is None despite wavs existing
    separated, cached = run_separation(store, input_path, separate_fn)
    assert cached is False
    assert len(calls) == 1
