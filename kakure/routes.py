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
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter()


@router.get("/static/i18n.js")
async def i18n_js():
    return FileResponse(_STATIC_DIR / "i18n.js", media_type="text/javascript")


def _enum_options(enum_cls) -> list[dict]:
    return [{"value": e.value, "label": e.value} for e in enum_cls]


def _voice_options() -> list[str]:
    return [v.value for v in ChineseVoice]


def _mix_mode_options() -> list[dict]:
    return [{"value": m.value, "label": m.value} for m in MixMode]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = load_settings()

    pipeline_stages = [
        {"id": "asr", "icon": "microphone"},
        {"id": "translate", "icon": "language"},
        {"id": "tts", "icon": "speaker-wave"},
        {"id": "separate", "icon": "musical-note"},
        {"id": "mix", "icon": "adjustments-horizontal"},
        {"id": "export", "icon": "arrow-down-tray"},
    ]

    transcribe_stages = [
        {"id": "load", "icon": "archive-box"},
        {"id": "transcribe", "icon": "microphone"},
    ]

    ctx = {
        "request": request,
        "settings_json": json.dumps(_settings_to_dict(settings)),
        "pipeline_stages_json": json.dumps(pipeline_stages),
        "transcribe_stages_json": json.dumps(transcribe_stages),
        "mix_modes": _mix_mode_options(),
        "chinese_voices_json": json.dumps(_voice_options()),
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
