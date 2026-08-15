"""FastAPI module - REST API and SSE endpoints for Kakure pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import queue as sync_queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from kakure.checkpoint import (
    CheckpointStore,
    clear_all_checkpoints,
    run_asr,
    run_separation,
    run_translation,
    run_tts,
)
from kakure.config import (
    Settings,
    _settings_to_dict,
    apply_model_env,
    load_settings,
    output_dir_path,
    save_settings,
)
from kakure.models import (
    delete_model as delete_model_from_cache,
)
from kakure.models import (
    download_model as download_model_to_cache,
)
from kakure.models import (
    model_status as list_models_status,
)
from kakure.srt import srt_path as srt_output_path
from kakure.srt import write_srt

logger = logging.getLogger(__name__)

# Point all model downloads at the unified model directory when model_dir is
# set in kakure.toml (portable / integrated-package mode). Runs at import
# time, before any model library (huggingface_hub, torch, ...) is imported.
apply_model_env(load_settings())

app = FastAPI(title="Kakure API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Friendly error mapping
# ---------------------------------------------------------------------------


def _friendly_error(exc: BaseException) -> str:
    """Map common failure modes to human-readable, actionable messages."""
    name = exc.__class__.__name__
    module = exc.__class__.__module__
    text = str(exc)

    if isinstance(exc, ModuleNotFoundError):
        if exc.name == "demucs":
            return (
                "Vocal separation requires Demucs, which is not installed.\n"
                'Install it with: pip install -e ".[demucs]"'
            )
        if exc.name == "indextts" or (exc.name or "").startswith("indextts."):
            return (
                "The IndexTTS-2.5 backend is not installed.\n"
                "Install it by running `kakure install-indextts` (git clone + "
                "uv sync --all-extras; requires an NVIDIA GPU)."
            )
        if exc.name in ("torch", "torchaudio", "transformers"):
            return (
                "The kotoba-whisper ASR backend needs PyTorch/Transformers, "
                "which are not installed.\n"
                'Install them with: pip install -e ".[kotoba]"'
            )

    if isinstance(exc, FileNotFoundError) and any(
        s in text.lower() for s in ("ffmpeg", "avprobe", "winerror 2", "errno 2")
    ):
        return (
            "ffmpeg was not found, and Kakure needs it to process audio.\n"
            "Download ffmpeg from https://ffmpeg.org/download.html, add it to your PATH, "
            "then restart Kakure."
        )

    if module.startswith("openai"):
        if name == "AuthenticationError":
            return "OpenAI rejected your API key. Open the Settings tab and check the key."
        if name == "RateLimitError":
            return "OpenAI rate limit hit — please wait a moment and try again."
        if name == "APIConnectionError":
            return (
                "Could not reach the OpenAI servers. Check your internet connection and try again."
            )

    if "api key required" in text.lower():
        return (
            "A translation API key is missing. Open the Settings tab and add it, "
            "or set the matching field in kakure.toml."
        )

    return text


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Tracks a single pipeline job."""

    id: str
    status: str = "pending"
    events: sync_queue.Queue[dict[str, Any]] = field(default_factory=sync_queue.Queue)
    result: dict[str, Any] | None = None
    error: str | None = None
    _temp_input: Path | None = None
    original_name: str | None = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _create_job() -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _push_event(job: Job, event: dict[str, Any]) -> None:
    job.events.put_nowait(event)


# ---------------------------------------------------------------------------
# Pipeline runner (background thread)
# ---------------------------------------------------------------------------


def _srt_base(job: Job, input_path: Path, settings: Settings) -> tuple[Path, str]:
    """Return ``(output_dir, stem)`` for generated SRT files.

    When ``output_dir`` is set, SRTs are written there using the original
    uploaded filename stem; otherwise they go next to the input temp file
    using its (job-id) stem to avoid collisions between concurrent jobs.
    """
    if settings.output_dir:
        output_dir = output_dir_path(settings, input_path)
        stem = Path(job.original_name).stem if job.original_name else input_path.stem
    else:
        output_dir = input_path.parent
        stem = input_path.stem
    return output_dir, stem


def _run_pipeline_job(job: Job, input_path: Path, settings: Settings) -> None:
    try:
        job.status = "running"
        srt_dir, srt_stem = _srt_base(job, input_path, settings)
        ckpt = CheckpointStore(input_path, settings)

        # --- Stage: ASR ---
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "asr",
                "status": "running",
                "message": "Transcribing Japanese audio...",
            },
        )

        from kakure.asr import create_asr_processor

        _t0 = time.perf_counter()
        asr = create_asr_processor(settings)
        transcription, asr_cached = run_asr(ckpt, lambda: asr.transcribe(input_path))
        if not transcription.segments:
            _push_event(
                job,
                {
                    "type": "error",
                    "stage": "asr",
                    "message": "No speech segments detected in the audio file.",
                },
            )
            job.status = "failed"
            job.error = "No speech segments detected"
            return

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "asr",
                "status": "completed",
                "cached": asr_cached,
                "segments": [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "original_text": seg.text,
                        "translated_text": "",
                    }
                    for seg in transcription.segments
                ],
                "segments_count": len(transcription.segments),
                "language": transcription.language,
                "duration": transcription.duration,
                "elapsed_seconds": time.perf_counter() - _t0,
            },
        )

        # Export Japanese subtitles right after ASR
        srt_ja_path = srt_output_path(srt_stem, "ja", srt_dir)
        write_srt(transcription.segments, srt_ja_path)

        # --- Stage: Translation ---
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "translate",
                "status": "running",
                "message": "Translating to Chinese...",
            },
        )

        from kakure.translator import TranslatedSegment, Translator

        _t0 = time.perf_counter()

        def _translate(segments):
            translated = [
                TranslatedSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    original_text=seg.text,
                    translated_text="",
                )
                for seg in segments
            ]
            return Translator(settings).translate_segments(translated)

        translated, tr_cached = run_translation(ckpt, transcription.segments, _translate)

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "translate",
                "status": "completed",
                "cached": tr_cached,
                "elapsed_seconds": time.perf_counter() - _t0,
                "segments": [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "original_text": seg.original_text,
                        "translated_text": seg.translated_text,
                    }
                    for seg in translated
                ],
            },
        )

        # Export Chinese subtitles right after translation
        srt_zh_path = srt_output_path(srt_stem, "zh", srt_dir)
        write_srt(translated, srt_zh_path, text=lambda seg: seg.translated_text)

        # --- Stage: TTS ---
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "tts",
                "status": "running",
                "message": "Generating Chinese voice...",
            },
        )

        from kakure.tts import create_tts_processor

        _t0 = time.perf_counter()
        segment_dicts = [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "original_text": seg.original_text,
                "translated_text": seg.translated_text,
            }
            for seg in translated
        ]
        tts = create_tts_processor(settings)
        tts_dicts, tts_reused = run_tts(
            ckpt,
            segment_dicts,
            lambda needed, outdir: [
                {
                    "segment_id": r.segment_id,
                    "audio_path": str(r.audio_path),
                    "duration_ms": r.duration_ms,
                }
                for r in tts.generate_sync(needed, output_dir=outdir)
            ],
        )

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "tts",
                "status": "completed",
                "cached": tts_reused > 0,
                "elapsed_seconds": time.perf_counter() - _t0,
                "segments": len(tts_dicts),
            },
        )

        # --- Stage: Vocal separation (optional) ---
        separated = None
        if settings.separate_vocals:
            _push_event(
                job,
                {
                    "type": "progress",
                    "stage": "separate",
                    "status": "running",
                    "message": "Separating vocals from background...",
                },
            )

            from kakure.separator import VocalSeparator

            _t0 = time.perf_counter()
            separator = VocalSeparator(settings)
            separated, sep_cached = run_separation(
                ckpt,
                input_path,
                lambda outdir: separator.separate(input_path, output_dir=outdir),
            )

            _push_event(
                job,
                {
                    "type": "progress",
                    "stage": "separate",
                    "status": "completed",
                    "cached": sep_cached,
                    "elapsed_seconds": time.perf_counter() - _t0,
                },
            )
        else:
            _push_event(
                job,
                {
                    "type": "progress",
                    "stage": "separate",
                    "status": "skipped",
                },
            )

        # --- Stage: Mixing ---
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "mix",
                "status": "running",
                "message": "Mixing audio tracks...",
            },
        )

        from pydub import AudioSegment

        from kakure.mixer import AudioMixer, MixInput

        original_audio = AudioSegment.from_file(str(input_path))
        _t0 = time.perf_counter()
        mix_input = MixInput(
            original_audio=original_audio,
            segments=segment_dicts,
            tts_results=tts_dicts,
            separated=separated,
        )
        mixer = AudioMixer(settings)
        mixed_audio = mixer.mix(mix_input)

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "mix",
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - _t0,
            },
        )

        # --- Stage: Export ---
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "export",
                "status": "running",
                "message": "Exporting final audio...",
            },
        )

        output_stem = (
            Path(job.original_name).stem
            if settings.output_dir and job.original_name
            else input_path.stem
        )
        output_path = output_dir_path(settings, input_path) / (
            f"{output_stem}_bilingual.{settings.output_format}"
        )
        _t0 = time.perf_counter()
        mixer.export(
            mixed_audio,
            output_path,
            format=settings.output_format,
            bitrate=settings.output_bitrate,
            sample_rate=settings.output_sample_rate,
        )

        srt_path = srt_output_path(srt_stem, "bilingual", srt_dir)
        write_srt(
            translated,
            srt_path,
            text=lambda seg: f"{seg.original_text}\n{seg.translated_text}",
        )
        _push_event(
            job,
            {
                "type": "progress",
                "stage": "export",
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - _t0,
            },
        )

        duration = len(mixed_audio) / 1000.0
        segments_data = [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "original_text": seg.original_text,
                "translated_text": seg.translated_text,
            }
            for seg in translated
        ]

        job.result = {
            "output_path": str(output_path),
            "srt_path": str(srt_path),
            "srt_ja_path": str(srt_ja_path),
            "srt_zh_path": str(srt_zh_path),
            "duration": duration,
            "segments": segments_data,
            "vocals_separated": separated is not None,
        }
        job.status = "completed"

        _push_event(
            job,
            {
                "type": "finished",
                **job.result,
            },
        )

    except Exception as e:
        logger.exception("Pipeline job %s failed", job.id)
        job.status = "failed"
        job.error = _friendly_error(e)
        _push_event(job, {"type": "error", "message": job.error})


# ---------------------------------------------------------------------------
# Model download runner (background thread)
# ---------------------------------------------------------------------------


def _run_download_job(job: Job, repo_id: str) -> None:
    """Download a model in the background, streaming progress via the job queue."""
    try:
        job.status = "running"

        def _on_progress(done: int, total: int) -> None:
            percent = round(done / total * 100, 1) if total else 0.0
            _push_event(
                job,
                {
                    "type": "progress",
                    "stage": "download",
                    "repo_id": repo_id,
                    "done": done,
                    "total": total,
                    "percent": percent,
                },
            )

        download_model_to_cache(repo_id, on_progress=_on_progress)
        job.status = "completed"
        job.result = {"repo_id": repo_id}
        _push_event(job, {"type": "finished", "repo_id": repo_id})
    except Exception as e:
        logger.exception("Model download job %s failed for %s", job.id, repo_id)
        job.status = "failed"
        job.error = _friendly_error(e)
        _push_event(job, {"type": "error", "repo_id": repo_id, "message": job.error})


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


async def _sse_stream(job: Job):
    loop = asyncio.get_running_loop()
    while True:
        try:
            event = await loop.run_in_executor(None, lambda: job.events.get(timeout=2.0))
        except sync_queue.Empty:
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            continue
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") in ("finished", "error"):
            break


# ---------------------------------------------------------------------------
# Process endpoints
# ---------------------------------------------------------------------------


@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    settings = load_settings()
    tmp_dir = Path(settings.temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    job = _create_job()
    job.original_name = file.filename
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    input_path = tmp_dir / f"kakure_input_{job.id}{suffix}"
    input_path.write_bytes(await file.read())
    job._temp_input = input_path

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job, input_path, settings),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job.id})


@app.get("/api/process/{job_id}")
async def process_progress(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    return StreamingResponse(
        _sse_stream(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/process/{job_id}/result")
async def process_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "pending" or job.status == "running":
        raise HTTPException(409, "Job not yet complete")
    if job.status == "failed":
        raise HTTPException(500, job.error or "Job failed")

    assert job.result is not None
    output_path = Path(job.result["output_path"])
    media_type = "audio/mpeg" if output_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(output_path, media_type=media_type, filename=output_path.name)


def _srt_response(job: Job, key: str, kind: str) -> FileResponse:
    """Serve a stored SRT file with a human-friendly download name."""
    assert job.result is not None
    srt_file = Path(job.result[key])
    if job.original_name:
        stem = Path(job.original_name).stem
        filename = f"{stem}_{kind}.srt"
    else:
        filename = srt_file.name
    return FileResponse(srt_file, media_type="text/plain", filename=filename)


@app.get("/api/process/{job_id}/srt")
async def process_srt(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, "Job not yet complete")

    return _srt_response(job, "srt_path", "bilingual")


@app.get("/api/process/{job_id}/srt_ja")
async def process_srt_ja(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, "Job not yet complete")

    return _srt_response(job, "srt_ja_path", "ja")


@app.get("/api/process/{job_id}/srt_zh")
async def process_srt_zh(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, "Job not yet complete")

    return _srt_response(job, "srt_zh_path", "zh")


@app.get("/api/process/{job_id}/segments")
async def process_segments(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, "Job not yet complete")

    assert job.result is not None
    return JSONResponse(job.result.get("segments", []))


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "error": job.error,
            "has_result": job.result is not None,
        }
    )


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def get_settings():
    settings = load_settings()
    return JSONResponse(_settings_to_dict(settings))


@app.put("/api/settings")
async def update_settings(body: dict[str, Any]):
    try:
        current = load_settings()
        updated_data = _settings_to_dict(current)
        updated_data.update({k: v for k, v in body.items() if k in updated_data})
        new_settings = Settings(**updated_data)
        save_settings(new_settings)
        return JSONResponse({"status": "ok", "settings": _settings_to_dict(new_settings)})
    except Exception as e:
        raise HTTPException(400, f"Invalid settings: {e}")


@app.post("/api/settings/reset")
async def reset_settings():
    try:
        defaults = Settings()
        save_settings(defaults)
        return JSONResponse({"status": "ok", "settings": _settings_to_dict(defaults)})
    except Exception as e:
        raise HTTPException(500, f"Failed to reset settings: {e}")


@app.post("/api/checkpoints/clear")
async def clear_checkpoints():
    try:
        cleared = clear_all_checkpoints(load_settings())
        return JSONResponse({"status": "ok", "cleared": cleared})
    except Exception as e:
        raise HTTPException(500, f"Failed to clear checkpoints: {e}")


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------


@app.get("/api/models")
async def models_status():
    """Return the model catalog annotated with install status and sizes."""
    try:
        return JSONResponse(list_models_status())
    except Exception as e:
        raise HTTPException(500, f"Failed to inspect models: {e}")


@app.post("/api/models/download")
async def models_download(body: dict[str, Any]):
    """Start a background download of a model, returning a job to stream from."""
    repo_id = body.get("repo_id")
    if not repo_id:
        raise HTTPException(400, "Missing 'repo_id'")

    job = _create_job()
    thread = threading.Thread(
        target=_run_download_job,
        args=(job, repo_id),
        daemon=True,
    )
    thread.start()
    return JSONResponse({"job_id": job.id})


@app.get("/api/models/download/{job_id}")
async def models_download_progress(job_id: str):
    """SSE stream of model download progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    return StreamingResponse(
        _sse_stream(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/models/delete")
async def models_delete(body: dict[str, Any]):
    """Remove a model from the local HuggingFace cache."""
    repo_id = body.get("repo_id")
    if not repo_id:
        raise HTTPException(400, "Missing 'repo_id'")
    try:
        freed = delete_model_from_cache(repo_id)
        return JSONResponse({"status": "ok", "freed": freed})
    except Exception as e:
        raise HTTPException(500, f"Failed to delete model: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok", "ffmpeg": shutil.which("ffmpeg") is not None})


# ---------------------------------------------------------------------------
# Mount web UI routes
# ---------------------------------------------------------------------------

from kakure.routes import router as web_router  # noqa: E402

app.include_router(web_router)
