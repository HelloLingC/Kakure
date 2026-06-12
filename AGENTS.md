# AGENTS.md

## Project Overview

Kakure is a CLI tool that translates Japanese ASMR voice audio into bilingual audio by overlaying a Chinese voice track. Pipeline: ASR (faster-whisper or kotoba-whisper) → Translation (OpenAI/DeepL) → TTS (edge-tts or IndexTTS) → Vocal separation (Demucs, optional) → Audio mixing (pydub).

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

## Commands

```bash
# Full pipeline (default: faster-whisper ASR, edge-tts TTS)
kakure process input.mp3 -m overlay -o output.mp3

# Full pipeline with kotoba-whisper ASR
kakure process input.mp3 --asr-backend kotoba-whisper -o output.mp3

# Full pipeline with IndexTTS (voice cloning from reference audio)
kakure process input.mp3 --tts-backend indextts --reference-audio voice_sample.wav -o output.mp3

# Process with vocal separation for cleaner mixing
kakure process input.mp3 --separate-vocals -o output.mp3

# ASR only (no translation, no API key needed)
kakure transcribe input.mp3

# ASR only with kotoba-whisper
kakure transcribe input.mp3 --asr-backend kotoba-whisper

# List Chinese TTS voices (network call to Microsoft)
kakure voices

# Show current config
kakure config
```

## Environment

Required env vars (set in `.env` or exported):
- `OPENAI_API_KEY` — required when `KAKURE_TRANSLATION_BACKEND=openai` (default)
- `DEEPL_API_KEY` — required when `KAKURE_TRANSLATION_BACKEND=deepl`

All settings use `KAKURE_` prefix (e.g., `KAKURE_WHISPER_MODEL`, `KAKURE_MIX_MODE`). See `.env.example` for full list.

## Architecture

Single-package layout: `kakure/` with these modules:

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic Settings, all enums (ASRBackend, MixMode, TranslationBackend, WhisperModelSize, KotobaWhisperModel, ChineseVoice, TTSBackend, DemucsModel) |
| `asr.py` | `BaseASRProcessor` (ABC) → `ASRProcessor` (faster-whisper) or `KotobaWhisperProcessor` (HuggingFace Transformers). Factory: `create_asr_processor(settings)`. Returns `TranscriptionResult` with `Segment`/`Word` dataclasses |
| `translator.py` | `Translator` → `OpenAITranslator` or `DeepLTranslator` — lazy-inits backend, uses ASMR-specific system prompt, passes rolling context of last 3 segments |
| `tts.py` | `BaseTTSProcessor` (ABC) → `EdgeTTSProcessor` (cloud, async) or `IndexTTSProcessor` (local GPU, voice cloning). Factory: `create_tts_processor(settings)`. Returns `TTSResult` dataclasses |
| `separator.py` | `VocalSeparator` — uses Demucs to split audio into vocals and background. Returns `SeparatedAudio` dataclass. Lazy-loads model on first use |
| `mixer.py` | `AudioMixer` — 4 modes: `dual` (stereo L=JP/R=CN), `overlay` (CN at -6dB with ducking), `sequential` (JP→gap→CN), `whisper` (CN at -15dB, no ducking). When `separated` is provided in `MixInput`, uses background + reduced vocals as base instead of original |
| `pipeline.py` | `Pipeline` — orchestrates ASR→Translation→TTS→(Separation)→Mixing→Export with Rich progress display. Uses factories to select backends |
| `cli.py` | Click CLI entry point, `kakure` command group |

Data flow: `Segment` (asr) → `TranslatedSegment` (translator) → `dict` segments + `TTSResult` dicts (tts) → `MixInput` (mixer, optionally with `SeparatedAudio`) → `AudioSegment` → exported file.

## ASR Backends

Two ASR backends are available, selected via `--asr-backend` or `KAKURE_ASR_BACKEND`:

| Backend | Install | Model sizes | Word timestamps | Japanese quality |
|---|---|---|---|---|
| `faster-whisper` (default) | Core deps | tiny/base/small/medium/large-v3/distil-large-v3 | ✅ Yes | Good (multilingual) |
| `kotoba-whisper` | `pip install -e ".[kotoba]"` | v2.0/v2.1(+punct)/v2.2(+diarization) | ❌ No | Better CER (Japanese-specific) |

Key differences:
- **kotoba-whisper** uses HuggingFace Transformers pipeline, not CTranslate2. Lazy-loaded on first use.
- **kotoba-whisper** `Segment.words` is always empty — only segment-level timestamps.
- **kotoba-whisper v2.2** requires `pyannote.audio` and accepting model terms on HuggingFace.

## TTS Backends

Two TTS backends are available, selected via `--tts-backend` or `KAKURE_TTS_BACKEND`:

| Backend | Install | Voice source | Output format | GPU required | Offline |
|---|---|---|---|---|---|
| `edge-tts` (default) | Core deps | Pre-built Chinese voices | MP3 | ❌ No | ❌ No |
| `indextts` | `pip install -e ".[indextts]"` | Voice cloning from reference audio | WAV | ✅ Yes (CUDA) | ✅ Yes |

Key differences:
- **IndexTTS** requires a reference audio file (`--reference-audio` or `KAKURE_INDEXTTS_REFERENCE_AUDIO`) — a 3-10 second sample of the desired voice.
- **IndexTTS** is synchronous (no async); **edge-tts** is async with `generate_sync()` wrapper.
- **IndexTTS** outputs WAV files; **edge-tts** outputs MP3. The mixer handles both via pydub.
- **IndexTTS** model is ~6GB, auto-downloaded from HuggingFace on first use.
- **IndexTTS** lazy-loads the model on first `infer()` call, not at construction.

## Vocal Separation (Demucs)

Optional feature enabled via `--separate-vocals` or `KAKURE_SEPARATE_VOCALS=true`. Uses Demucs to split the original audio into vocals and background (drums+bass+other), then:

- Background plays at `KAKURE_BACKGROUND_VOLUME_DB` (default: 0dB, unchanged)
- Original vocals play at `KAKURE_VOCALS_VOLUME_DB` (default: -6dB, reduced)
- Chinese TTS overlays on top at the configured mix volume

This creates a cleaner bilingual mix because the Chinese voice doesn't compete with the original Japanese vocals. Available models: `htdemucs` (default), `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`.

Install: `pip install -e ".[demucs]"`. Model downloads ~2-6GB on first use.

## Key Gotchas

- **Whisper model downloads on first run** — `large-v3` is ~3GB. Use `--model tiny` or `--model small` for quick testing.
- **kotoba-whisper downloads on first run** — models are ~1.5GB, fetched from HuggingFace Hub.
- **IndexTTS downloads on first run** — model is ~6GB, fetched from HuggingFace Hub. Requires NVIDIA GPU with CUDA.
- **IndexTTS requires reference audio** — must provide `--reference-audio` or `KAKURE_INDEXTTS_REFERENCE_AUDIO` when using `--tts-backend indextts`.
- **Demucs model downloads on first run** — `htdemucs` is ~2GB. Use `--demucs-device cuda` for GPU acceleration.
- **edge-tts is async** — `EdgeTTSProcessor.generate()` is async; `generate_sync()` wraps it with `asyncio.run()`. Don't nest inside another event loop.
- **Translation backends are lazy** — API keys are only validated when `Translator.backend` property is first accessed, not at construction.
- **Config mutation** — CLI `process` command mutates the `Settings` object directly from CLI args. Tests or programmatic use should construct `Settings` explicitly rather than relying on `get_settings()`.
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