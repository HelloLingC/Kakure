"""Web routes — serves the HTMX+Alpine single-page UI."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from kakure.config import (
    ASRBackend,
    ChineseVoice,
    DemucsModel,
    KotobaWhisperModel,
    MixMode,
    TranslationBackend,
    TTSBackend,
    WhisperModelSize,
    _settings_to_dict,
    load_settings,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Reference audio folder. In a packaged build __file__ lives inside site-
# packages, so prefer a references/ folder next to the working directory
# (the package root when launched via start-kakure.bat) before falling back
# to the dev-tree location.
_REFERENCE_DIR = (
    Path("references")
    if Path("references").is_dir()
    else _PROJECT_ROOT / "references"
)
_REFERENCE_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter()


@router.get("/static/i18n.js")
async def i18n_js():
    return FileResponse(
        _STATIC_DIR / "i18n.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/static/vendor/{filename}")
async def vendor_js(filename: str):
    vendor_dir = (_STATIC_DIR / "vendor").resolve()
    target = (vendor_dir / filename).resolve()
    if vendor_dir not in target.parents or not target.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(
        target,
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


def _enum_options(enum_cls) -> list[dict]:
    return [{"value": e.value, "label": e.value} for e in enum_cls]


def _voice_options() -> list[str]:
    return [v.value for v in ChineseVoice]


def _mix_mode_options() -> list[dict]:
    return [{"value": m.value, "label": m.value} for m in MixMode]


def _reference_audio_options() -> list[str]:
    """List audio files available in the project-root references/ folder."""
    if not _REFERENCE_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in _REFERENCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _REFERENCE_EXTS
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = load_settings()

    # Cache-buster for static assets: any change to the file changes the URL.
    i18n_version = _STATIC_DIR.joinpath("i18n.js").stat().st_mtime_ns

    pipeline_stages = [
        {"id": "asr", "icon": "microphone"},
        {"id": "translate", "icon": "language"},
        {"id": "tts", "icon": "speaker-wave"},
        {"id": "separate", "icon": "musical-note"},
        {"id": "mix", "icon": "adjustments-horizontal"},
        {"id": "export", "icon": "arrow-down-tray"},
    ]

    ctx = {
        "request": request,
        "i18n_version": i18n_version,
        "settings_json": json.dumps(_settings_to_dict(settings)),
        "pipeline_stages_json": json.dumps(pipeline_stages),
        "mix_modes": _mix_mode_options(),
        "chinese_voices_json": json.dumps(_voice_options()),
        "references_audio_json": json.dumps(_reference_audio_options()),
        "asr_backends": _enum_options(ASRBackend),
        "whisper_models": _enum_options(WhisperModelSize),
        "kotoba_models": _enum_options(KotobaWhisperModel),
        "tts_backends": _enum_options(TTSBackend),
        "demucs_models": _enum_options(DemucsModel),
        "translation_backends": _enum_options(TranslationBackend),
        "output_formats": ["mp3", "wav", "m4a", "flac", "ogg"],
        "bitrates": ["128k", "192k", "256k", "320k"],
    }
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)
