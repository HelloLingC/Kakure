"""FastAPI module - REST API and SSE endpoints for Kakure pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import queue as sync_queue
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from kakure.config import Settings, _settings_to_dict, load_settings, save_settings

logger = logging.getLogger(__name__)

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
        if exc.name == "indextts":
            return (
                "The IndexTTS backend is not installed.\n"
                'Install it with: pip install -e ".[indextts]" (requires an NVIDIA GPU).'
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
    """Tracks a single pipeline or transcription job."""

    id: str
    status: str = "pending"
    events: sync_queue.Queue[dict[str, Any]] = field(default_factory=sync_queue.Queue)
    result: dict[str, Any] | None = None
    error: str | None = None
    _temp_input: Path | None = None


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
# SRT export helper
# ---------------------------------------------------------------------------


def _write_srt(segments: list, path: Path) -> None:
    lines: list[str] = []
    for seg in segments:
        start_h, start_rem = divmod(seg.start, 3600)
        start_m, start_s = divmod(start_rem, 60)
        start_ms = int((seg.start - int(seg.start)) * 1000)

        end_h, end_rem = divmod(seg.end, 3600)
        end_m, end_s = divmod(end_rem, 60)
        end_ms = int((seg.end - int(seg.end)) * 1000)

        lines.append(str(seg.id + 1))
        lines.append(
            f"{int(start_h):02d}:{int(start_m):02d}:{int(start_s):02d},{start_ms:03d}"
            f" --> "
            f"{int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d},{end_ms:03d}"
        )
        lines.append(seg.translated_text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline runner (background thread)
# ---------------------------------------------------------------------------


def _run_pipeline_job(job: Job, input_path: Path, settings: Settings) -> None:
    try:
        job.status = "running"

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

        asr = create_asr_processor(settings)
        transcription = asr.transcribe(input_path)
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
                "segments": len(transcription.segments),
                "language": transcription.language,
                "duration": transcription.duration,
            },
        )

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

        translated = [
            TranslatedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text="",
            )
            for seg in transcription.segments
        ]
        translator = Translator(settings)
        translated = translator.translate_segments(translated)

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "translate",
                "status": "completed",
            },
        )

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
        tts_results = tts.generate_sync(segment_dicts)
        tts_dicts = [
            {
                "segment_id": r.segment_id,
                "audio_path": str(r.audio_path),
                "duration_ms": r.duration_ms,
            }
            for r in tts_results
        ]

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "tts",
                "status": "completed",
                "segments": len(tts_results),
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

            separator = VocalSeparator(settings)
            separated = separator.separate(input_path)

            _push_event(
                job,
                {
                    "type": "progress",
                    "stage": "separate",
                    "status": "completed",
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

        output_path = input_path.parent / f"{input_path.stem}_bilingual.{settings.output_format}"
        mixer.export(
            mixed_audio,
            output_path,
            format=settings.output_format,
            bitrate=settings.output_bitrate,
            sample_rate=settings.output_sample_rate,
        )

        srt_path = input_path.parent / f"{input_path.stem}_bilingual.srt"
        _write_srt(translated, srt_path)

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


def _run_transcribe_job(job: Job, input_path: Path, settings: Settings) -> None:
    try:
        job.status = "running"

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "load",
                "status": "running",
                "message": "Loading ASR model...",
            },
        )

        from kakure.asr import create_asr_processor

        asr = create_asr_processor(settings)

        _push_event(
            job,
            {"type": "progress", "stage": "load", "status": "completed"},
        )

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "transcribe",
                "status": "running",
                "message": "Transcribing...",
            },
        )

        result = asr.transcribe(input_path)

        segments_data = [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in result.segments
        ]

        job.result = {
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "segments": segments_data,
        }
        job.status = "completed"

        _push_event(
            job,
            {
                "type": "progress",
                "stage": "transcribe",
                "status": "completed",
                "segments": len(result.segments),
            },
        )
        _push_event(job, {"type": "finished", **job.result})

    except Exception as e:
        logger.exception("Transcription job %s failed", job.id)
        job.status = "failed"
        job.error = _friendly_error(e)
        _push_event(job, {"type": "error", "message": job.error})


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


@app.get("/api/process/{job_id}/srt")
async def process_srt(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, "Job not yet complete")

    assert job.result is not None
    srt_path = Path(job.result["srt_path"])
    return FileResponse(srt_path, media_type="text/plain", filename=srt_path.name)


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
# Transcribe endpoints
# ---------------------------------------------------------------------------


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    settings = load_settings()
    tmp_dir = Path(settings.temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    job = _create_job()
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    input_path = tmp_dir / f"kakure_input_{job.id}{suffix}"
    input_path.write_bytes(await file.read())
    job._temp_input = input_path

    thread = threading.Thread(
        target=_run_transcribe_job,
        args=(job, input_path, settings),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job.id})


@app.get("/api/transcribe/{job_id}")
async def transcribe_progress(job_id: str):
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


@app.get("/api/transcribe/{job_id}/result")
async def transcribe_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "pending" or job.status == "running":
        raise HTTPException(409, "Job not yet complete")
    if job.status == "failed":
        raise HTTPException(500, job.error or "Job failed")

    return JSONResponse(job.result)


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
