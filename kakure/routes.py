"""Web routes — serves the HTMX+Alpine single-page UI."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
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
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter()


def _enum_options(enum_cls) -> list[dict]:
    return [{"value": e.value, "label": e.value} for e in enum_cls]


def _voice_options() -> list[dict]:
    descriptions = {
        ChineseVoice.XIAOXIAO: "Female, warm",
        ChineseVoice.XIAOYI: "Female, gentle",
        ChineseVoice.YUNJIAN: "Male, calm",
        ChineseVoice.YUNXI: "Male, warm",
        ChineseVoice.YUNXIA: "Female, sweet",
        ChineseVoice.YUNYANG: "Male, news anchor",
    }
    return [
        {"value": v.value, "label": f"{v.value} ({descriptions.get(v, '')})".strip()}
        for v in ChineseVoice
    ]


_MIX_MODE_DESCRIPTIONS = {
    "dual": "Japanese left channel, Chinese right channel",
    "overlay": "Chinese voice overlaid at lower volume (-6dB) with ducking",
    "sequential": "Japanese segment, silence gap, then Chinese translation — output is longer than input",
    "whisper": "Chinese voice at very low volume (-15dB, no ducking) — subtle background hint",
    "spatial": "Cross-panned stereo: Japanese stronger left, Chinese stronger right",
}


def _mix_mode_options() -> list[dict]:
    return [
        {
            "value": m.value,
            "label": m.value,
            "desc": _MIX_MODE_DESCRIPTIONS.get(m.value, ""),
        }
        for m in MixMode
    ]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = load_settings()

    pipeline_stages = [
        {"id": "asr", "name": "ASR", "icon": "🎙️", "desc": "Transcribe Japanese"},
        {"id": "translate", "name": "Translation", "icon": "🌐", "desc": "JP → CN"},
        {"id": "tts", "name": "TTS", "icon": "🔊", "desc": "Generate Chinese voice"},
        {"id": "separate", "name": "Vocal Sep.", "icon": "🎵", "desc": "Separate vocals"},
        {"id": "mix", "name": "Mixing", "icon": "🎛️", "desc": "Mix audio tracks"},
        {"id": "export", "name": "Export", "icon": "💾", "desc": "Save output file"},
    ]

    transcribe_stages = [
        {"id": "load", "name": "Load Model", "icon": "📦", "desc": "Load ASR model"},
        {"id": "transcribe", "name": "Transcribe", "icon": "🎙️", "desc": "Speech recognition"},
    ]

    ctx = {
        "request": request,
        "settings_json": json.dumps(_settings_to_dict(settings)),
        "pipeline_stages_json": json.dumps(pipeline_stages),
        "transcribe_stages_json": json.dumps(transcribe_stages),
        "mix_modes": _mix_mode_options(),
        "mix_mode_descriptions_json": json.dumps(_MIX_MODE_DESCRIPTIONS),
        "asr_backends": _enum_options(ASRBackend),
        "whisper_models": _enum_options(WhisperModelSize),
        "kotoba_models": _enum_options(KotobaWhisperModel),
        "tts_backends": _enum_options(TTSBackend),
        "chinese_voices": _voice_options(),
        "demucs_models": _enum_options(DemucsModel),
        "translation_backends": _enum_options(TranslationBackend),
        "output_formats": ["mp3", "wav", "m4a", "flac", "ogg"],
        "bitrates": ["128k", "192k", "256k", "320k"],
    }
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)
