# AGENTS.md

## Project Overview

Kakure is a web-based tool that translates Japanese ASMR voice audio into bilingual audio by overlaying a Chinese voice track. Pipeline: ASR (faster-whisper or kotoba-whisper) → Translation (OpenAI) → TTS (edge-tts or IndexTTS) → Vocal separation (Demucs, optional) → Audio mixing (pydub).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For kotoba-whisper ASR backend (optional):
pip install -e ".[kotoba]"

# For IndexTTS TTS backend (optional):
pip install -e ".[indextts]"

# For vocal separation with Demucs (optional):
pip install -e ".[demucs]"
```

Requires **ffmpeg** on PATH (pydub dependency for non-WAV formats).

## Usage

```bash
# Launch the web UI (default: http://127.0.0.1:7860, browser opens automatically)
kakure

# With options
kakure --port 8080          # custom port
kakure --host 0.0.0.0       # bind to all interfaces (LAN/remote access)
kakure --no-browser         # do not auto-open a browser window
kakure --reload             # auto-reload on code changes (development)
```

The web UI has three tabs:
- **Process** — Upload Japanese ASMR audio, configure pipeline options, run the full bilingual pipeline, and download the output
- **Transcribe** — Upload audio for ASR-only transcription (no translation, no API key needed)
- **Settings** — Edit and persist all configuration options. Changes are saved to `kakure.toml` and loaded as defaults for Process and Transcribe tabs.

## Configuration

Settings are stored in `kakure.toml` (in the current working directory). Copy `kakure.toml.example` to get started:

```bash
cp kakure.toml.example kakure.toml
# Edit kakure.toml with your API keys and preferences
```

Required settings:
- `openai_api_key` — required when `translation_backend = "openai"` (default)

All settings can also be changed in the web UI (Settings tab) and are persisted to `kakure.toml`.

## Architecture

Single-package layout: `kakure/` with these modules:

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic `Settings` model with TOML load/save (`load_settings`, `save_settings`, `get_settings`). All enums (ASRBackend, MixMode, TranslationBackend, WhisperModelSize, KotobaWhisperModel, ChineseVoice, TTSBackend, DemucsModel) |
| `asr.py` | `BaseASRProcessor` (ABC) → `ASRProcessor` (faster-whisper) or `KotobaWhisperProcessor` (HuggingFace Transformers). Factory: `create_asr_processor(settings)`. Returns `TranscriptionResult` with `Segment`/`Word` dataclasses |
| `translator.py` | `Translator` → `OpenAITranslator` — lazy-inits backend, uses ASMR-specific system prompt, passes rolling context of last 3 segments |
| `tts.py` | `BaseTTSProcessor` (ABC) → `EdgeTTSProcessor` (cloud, async) or `IndexTTSProcessor` (local GPU, voice cloning). Factory: `create_tts_processor(settings)`. Returns `TTSResult` dataclasses |
| `separator.py` | `VocalSeparator` — uses Demucs to split audio into vocals and background. Returns `SeparatedAudio` dataclass. Lazy-loads model on first use |
| `mixer.py` | `AudioMixer` — 5 modes: `dual` (stereo L=JP/R=CN), `overlay` (CN at -6dB with ducking), `sequential` (JP→gap→CN), `whisper` (CN at -15dB, no ducking), `spatial` (cross-panned stereo). When `separated` is provided in `MixInput`, uses background + reduced vocals as base instead of original |
| `pipeline.py` | `Pipeline` — orchestrates ASR→Translation→TTS→(Separation)→Mixing→Export. Uses factories to select backends |
| `api.py` | FastAPI application — REST endpoints (`/api/process`, `/api/transcribe`, `/api/settings`), SSE progress streaming, background job store. `_friendly_error()` maps common failures (missing ffmpeg, optional deps, OpenAI auth/network) to actionable messages shown in the UI. `/api/health` reports `ffmpeg` availability for the UI banner |
| `routes.py` | Web UI routes — serves the SPA (`GET /`) with Jinja2 templates, passes enum options and settings as context |
| `cli.py` | CLI entry point (`kakure` command) — launches uvicorn with configurable host/port/reload. Default host is `127.0.0.1`; auto-opens the browser unless `--no-browser` is passed |
| `templates/index.html` | Single-page web UI — HTMX+Alpine.js with Process, Transcribe, and Settings tabs. Tailwind CSS via CDN. Alpine handles SSE progress streaming, file uploads, form state, and conditional field visibility. The Settings tab uses subtabs (ASR/Translator/TTS/Mixing/Output) with expanding cards |

Data flow: `Segment` (asr) → `TranslatedSegment` (translator) → `dict` segments + `TTSResult` dicts (tts) → `MixInput` (mixer, optionally with `SeparatedAudio`) → `AudioSegment` → exported file.

## ASR Backends

Two ASR backends, selectable in the web UI or via `asr_backend` in `kakure.toml`:

| Backend | Install | Model sizes | Word timestamps | Japanese quality |
|---|---|---|---|---|
| `faster-whisper` (default) | Core deps | tiny/base/small/medium/large-v3/distil-large-v3 | ✅ Yes | Good (multilingual) |
| `kotoba-whisper` | `pip install -e ".[kotoba]"` | v2.0/v2.1(+punct)/v2.2(+diarization) | ❌ No | Better CER (Japanese-specific) |

Key differences:
- **kotoba-whisper** uses HuggingFace Transformers pipeline, not CTranslate2. Lazy-loaded on first use.
- **kotoba-whisper** `Segment.words` is always empty — only segment-level timestamps.
- **kotoba-whisper v2.2** requires `pyannote.audio` and accepting model terms on HuggingFace.

## TTS Backends

Two TTS backends, selectable in the web UI or via `tts_backend` in `kakure.toml`:

| Backend | Install | Voice source | Output format | GPU required | Offline |
|---|---|---|---|---|---|
| `edge-tts` (default) | Core deps | Pre-built Chinese voices | MP3 | ❌ No | ❌ No |
| `indextts` | `pip install -e ".[indextts]"` | Voice cloning from reference audio | WAV | ✅ Yes (CUDA) | ✅ Yes |

Key differences:
- **IndexTTS** requires a reference audio file (upload in web UI or set `indextts_reference_audio` in `kakure.toml`) — a 3-10 second sample of the desired voice.
- **IndexTTS** is synchronous (no async); **edge-tts** is async with `generate_sync()` wrapper.
- **IndexTTS** outputs WAV files; **edge-tts** outputs MP3. The mixer handles both via pydub.
- **IndexTTS** model is ~6GB, auto-downloaded from HuggingFace on first use.
- **IndexTTS** lazy-loads the model on first `infer()` call, not at construction.

## Vocal Separation (Demucs)

Optional feature enabled via the web UI or `separate_vocals = true` in `kakure.toml`. Uses Demucs to split the original audio into vocals and background (drums+bass+other), then:

- Background plays at `background_volume_db` (default: 0dB, unchanged)
- Original vocals play at `vocals_volume_db` (default: -6dB, reduced)
- Chinese TTS overlays on top at the configured mix volume

This creates a cleaner bilingual mix because the Chinese voice doesn't compete with the original Japanese vocals. Available models: `htdemucs` (default), `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`.

Install: `pip install -e ".[demucs]"`. Model downloads ~2-6GB on first use.

## Key Gotchas

- **Whisper model downloads on first run** — `large-v3` is ~3GB. Use `tiny` or `small` for quick testing.
- **kotoba-whisper downloads on first run** — models are ~1.5GB, fetched from HuggingFace Hub.
- **IndexTTS downloads on first run** — model is ~6GB, fetched from HuggingFace Hub. Requires NVIDIA GPU with CUDA.
- **IndexTTS requires reference audio** — must upload a reference audio file in the web UI or set `indextts_reference_audio` in `kakure.toml` when using IndexTTS.
- **Demucs model downloads on first run** — `htdemucs` is ~2GB. Use `demucs_device = "cuda"` for GPU acceleration.
- **edge-tts is async** — `EdgeTTSProcessor.generate()` is async; `generate_sync()` wraps it with `asyncio.run()`. Don't nest inside another event loop.
- **Translation backends are lazy** — API keys are only validated when `Translator.backend` property is first accessed, not at construction.
- **Config is TOML** — settings are stored in `kakure.toml`, not `.env`. Use the web UI Settings tab or edit the file directly.
- **pydub uses milliseconds** — all timing in pydub is ms. ASR timestamps are in seconds. Conversion happens in mixer via `int(seg["start"] * 1000)`.
- **TTS audio trimming** — if generated Chinese speech is longer than the Japanese segment, it gets trimmed (not time-stretched). This can cut off translations in overlay/dual/whisper modes.
- **Sequential mode extends duration** — unlike other modes, sequential inserts gaps and TTS audio, making the output longer than the input.

## Linting & Formatting

```bash
ruff check kakure/
ruff format kakure/
```

Config in `pyproject.toml`: target py310, line-length 100, rules E/F/I/N/W/UP.

## Tests

```bash
pytest
```

Test dir: `tests/` (configured in `pyproject.toml`). No tests exist yet. `pytest-asyncio` is installed with `asyncio_mode = "auto"`.