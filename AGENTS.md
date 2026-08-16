# AGENTS.md

## Project Overview

Kakure is a web-based tool that translates Japanese ASMR voice audio into bilingual audio by overlaying a Chinese voice track. Pipeline: ASR (faster-whisper or kotoba-whisper) → Translation (OpenAI) → TTS (audiocpp / audio.cpp) → Vocal separation (Demucs, optional) → Audio mixing (pydub).

## Setup

### Windows one-click install (non-technical users)

Non-technical Windows users should double-click `install.bat`. It auto-installs Python 3.12
if needed, then delegates the rest to `install.py` (stdlib-only, in the repo root), which
creates `.venv`, upgrades pip, installs deps with optional Tsinghua mirror, downloads
a shared (DLL) ffmpeg 8.1.2 build to `bin\ffmpeg` and registers its DLL directory for
torchcodec via a venv `.pth` (`os.add_dll_directory`), and generates `kakure.toml`. Users
then double-click
`start-kakure.bat` to launch (or accept the launch prompt at the end of the installer).
Batch files are ASCII + CRLF (no BOM), first line `@echo off` and second `chcp 936 >nul`
(retained for Windows Terminal compatibility); because the messages are pure ASCII, the
UTF-8/GBK mis-parse bug does not apply to them. `install.py` is UTF-8. See `.gitattributes`
`-text` so batch bytes survive git checkout unchanged.

### Manual install (developers)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For kotoba-whisper ASR backend (optional):
pip install -e ".[kotoba]"

# For vocal separation with Demucs (optional):
pip install -e ".[demucs]"
```

Requires **ffmpeg** on PATH (pydub dependency for non-WAV formats).

## Usage

```bash
# Launch the web UI (default: http://127.0.0.1:7530, browser opens automatically)
kakure

# With options
kakure --port 8080          # custom port
kakure --host 0.0.0.0       # bind to all interfaces (LAN/remote access)
kakure --no-browser         # do not auto-open a browser window
kakure --reload             # auto-reload on code changes (development)
```

The web UI has three tabs:
- **Process** — Upload Japanese ASMR audio, configure pipeline options, run the full bilingual pipeline, and download the output
- **Settings** — Edit and persist all configuration options. Changes are saved to `kakure.toml` and loaded as defaults for the Process tab.
- **Models** — Browse the Whisper and audiocpp TTS models Kakure uses, with per-model Download/Delete buttons, live download progress (SSE), and install status from the HuggingFace Hub cache.

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
| `config.py` | Pydantic `Settings` model with TOML load/save (`load_settings`, `save_settings`, `get_settings`). All enums (ASRBackend, MixMode, TranslationBackend, WhisperModelSize, KotobaWhisperModel, TTSBackend, DemucsModel). `apply_model_env()` routes all model downloads into `model_dir` (portable/整合包 mode) by setting `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` before any model library imports |
| `asr.py` | `BaseASRProcessor` (ABC) → `ASRProcessor` (faster-whisper) or `KotobaWhisperProcessor` (HuggingFace Transformers). Factory: `create_asr_processor(settings)`. Returns `TranscriptionResult` with `Segment`/`Word` dataclasses |
| `translator.py` | `Translator` → `OpenAITranslator` — lazy-inits backend, uses ASMR-specific system prompt. Splits segments into independent batches and dispatches them concurrently via `ThreadPoolExecutor` bounded by `translation_max_concurrency` (no rolling context) |
| `tts.py` | `BaseTTSProcessor` (ABC) → `AudioCppTTSProcessor` (drives the bundled `audiocpp_server.exe` / audio.cpp engine over its OpenAI-compatible REST API). Factory: `create_tts_processor(settings)`. Returns `TTSResult` dataclasses. The server is launched as a sidecar process and managed as a module-level singleton |
| `separator.py` | `VocalSeparator` — uses Demucs to split audio into vocals and background. Returns `SeparatedAudio` dataclass. Lazy-loads model on first use |
| `mixer.py` | `AudioMixer` — 5 modes: `dual` (stereo L=JP/R=CN), `overlay` (CN at -6dB with ducking), `sequential` (JP→gap→CN), `whisper` (CN at -15dB, no ducking), `spatial` (cross-panned stereo). When `separated` is provided in `MixInput`, uses background + reduced vocals as base instead of original |
| `checkpoint.py` | `CheckpointStore` — caches ASR/translation/TTS/separation results under `<checkpoint_dir>/<input-sha256>/` (default `<temp_dir>/checkpoints/`). Each stage stores a JSON record with a settings fingerprint + upstream artifact hash; changing settings or upstream data invalidates the stage (and downstream). Builder helpers (`run_asr`, `run_translation`, `run_tts`, `run_separation`) are shared by the CLI pipeline and the web API. TTS resumes per-segment via text-hash comparison; Demucs WAVs are reused when present |
| `pipeline.py` | `Pipeline` — orchestrates ASR→Translation→TTS→(Separation)→Mixing→Export. Uses factories to select backends |
| `api.py` | FastAPI application — REST endpoints (`/api/process`, `/api/settings`), SSE progress streaming, background job store. `_friendly_error()` maps common failures (missing ffmpeg, optional deps, OpenAI auth/network) to actionable messages shown in the UI. `/api/health` reports `ffmpeg` availability for the UI banner |
| `routes.py` | Web UI routes — serves the SPA (`GET /`) with Jinja2 templates, passes enum options and settings as context |
| `cli.py` | CLI entry point (`kakure` command) — launches uvicorn with configurable host/port/reload. Default host is `127.0.0.1`; auto-opens the browser unless `--no-browser` is passed |
| `models.py` | Model management — catalog of Whisper (faster-whisper + kotoba-whisper) and audiocpp TTS model repos, install-status/size reporting from the HuggingFace Hub cache (resolved to `model_dir` when set), background downloads with progress callbacks (`hf_hub_download` for single GGUF files, `snapshot_download` otherwise) and cache deletion |
| `templates/index.html` | Single-page web UI — HTMX+Alpine.js with Process, Settings, and Models tabs. Tailwind CSS via CDN. Alpine handles SSE progress streaming, file uploads, form state, and conditional field visibility. The Settings tab uses subtabs (ASR/Translator/TTS/Mixing/Output) with expanding cards. The Models tab lists cached/uncached models with Download/Delete and live progress bars |

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

## TTS Backend (audiocpp / audio.cpp)

Speech synthesis is performed by the **audiocpp** engine (the bundled `audiocpp_server.exe`, a Windows CUDA build of audio.cpp). It exposes an OpenAI-compatible REST API; Kakure launches it as a sidecar process, pins the TTS model via a generated `server.json`, and calls `POST /v1/audio/speech` per translated segment. Voice cloning is configured once through the server's `default_voice_preset` (reference audio + transcript), so each request only needs the input text.

Key points:
- Configured via `tts_backend = "audiocpp"` (the only TTS backend) plus `audiocpp_*` settings (`audiocpp_host`, `audiocpp_port`, `audiocpp_backend` cuda/cpu, `audiocpp_family`, `audiocpp_model`, `audiocpp_language`, `audiocpp_reference_audio`, `audiocpp_reference_text`, `audiocpp_speed`, `audiocpp_max_tokens`).
- `audiocpp_server.exe` is bundled under `./audiocpp`; set `audiocpp_exe` to override its path.
- `audiocpp_family` selects a model family (default `qwen3_tts`, which supports Chinese/Japanese and zero-shot voice cloning). When `audiocpp_model` is empty, the default GGUF for the family is auto-discovered from the HuggingFace cache.
- Output is always WAV; the mixer handles it via pydub.
- Requires an NVIDIA GPU with CUDA and the matching CUDA runtime DLLs for `audiocpp_server.exe`. The Qwen3-TTS GGUF (~1.7B, q8_0) is downloaded on first use via the Models tab.
- The sidecar server is a module-level singleton: it starts lazily on first TTS use, restarts only when its config signature changes, and is stopped on process exit (`atexit`).

## Vocal Separation (Demucs)

Optional feature enabled via the web UI or `separate_vocals = true` in `kakure.toml`. Uses Demucs to split the original audio into vocals and background (drums+bass+other), then:

- Background plays at `background_volume_db` (default: 0dB, unchanged)
- Original vocals play at `vocals_volume_db` (default: -6dB, reduced)
- Chinese TTS overlays on top at the configured mix volume

This creates a cleaner bilingual mix because the Chinese voice doesn't compete with the original Japanese vocals. Available models: `htdemucs` (default), `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`.

Install: `pip install -e ".[demucs]"`. Model downloads ~2-6GB on first use.

## Key Gotchas

- **Whisper model downloads on first run** — `large-v3` is ~3GB. Use `tiny` or `small` for quick testing. By default all models go to the HuggingFace Hub cache (`~/.cache/huggingface`); set `model_dir` (e.g. `models`) in `kakure.toml` to keep every model file inside the project folder — required for a portable 整合包 build. `apply_model_env()` (called from `cli.py` and `api.py` startup) sets `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` accordingly.
- **torchcodec needs a *shared* FFmpeg** — its `libtorchcodec_core*.dll` depends on `avcodec-*.dll` etc., which the Windows loader does not resolve from PATH (safe-DLL-search). The installer bundles FFmpeg 8.1.2 full-shared into `bin\ffmpeg\bin` and registers it for the venv via `os.add_dll_directory` in `.venv/Lib/site-packages/torchcodec_ffmpeg_path.pth`. A static `ffmpeg.exe` (e.g. choco's) is not enough, and FFmpeg 9 is too new for torchcodec 0.15.
- **kotoba-whisper downloads on first run** — models are ~1.5GB, fetched from HuggingFace Hub.
- **audiocpp TTS downloads on first run** — the Qwen3-TTS GGUF (~1.7B, q8_0) is fetched from HuggingFace Hub via the Models tab. Requires NVIDIA GPU with CUDA and the matching CUDA runtime DLLs for `audiocpp_server.exe`.
- **audiocpp voice cloning needs reference audio** — optionally upload a reference audio file in the web UI or set `audiocpp_reference_audio` (and `audiocpp_reference_text`) in `kakure.toml`. A 3-10 second sample of the desired voice works best.
- **Demucs model downloads on first run** — `htdemucs` is ~2GB. Use `demucs_device = "cuda"` for GPU acceleration.
- **Translation backends are lazy** — API keys are only validated when `Translator.backend` property is first accessed, not at construction.
- **Translation is context-free and concurrent** — batches are independent (no rolling context) and dispatched via `ThreadPoolExecutor` bounded by `translation_max_concurrency` (default 4). 1 = sequential. Higher values are faster but can hit provider rate limits; a failed batch falls back to per-segment requests.
- **Config is TOML** — settings are stored in `kakure.toml`, not `.env`. Use the web UI Settings tab or edit the file directly.
- **Checkpoints are on by default** — completed ASR/translation/TTS/Demucs stages are cached under `<temp_dir>/checkpoints/<input-sha256>/` and reused on re-runs. Disable via `enable_checkpoints = false`; clear all caches via the Settings tab button or `POST /api/checkpoints/clear`. Cached data is invalidated automatically when stage-affecting settings change (fingerprint) or the upstream data changes (hash chain).
- **TTS resumes per segment** — a segment's cached audio is reused only if its text hash still matches the current translation; otherwise it is regenerated. Both TTS processors skip existing non-empty segment files, so a hard crash mid-TTS resumes on the next run.
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