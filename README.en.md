# Kakure (隠れ)

[简体中文](README.md)

ASMR Japanese-to-Chinese bilingual voice overlay tool.

Translates Japanese ASMR voice audio into bilingual voice audio by overlaying a Chinese voice track into the original audio.

## Features

- **ASR**: Japanese speech recognition with timestamps — choose between faster-whisper (multilingual) or kotoba-whisper (Japanese-optimized)
- **Translation**: Japanese-to-Chinese translation (OpenAI GPT)
- **TTS**: Chinese voice generation — choose between edge-tts (cloud, pre-built voices) or IndexTTS (local GPU, voice cloning)
- **Vocal Separation**: Optional Demucs-based separation of vocals from background for cleaner mixing
- **Mixing**: Four mixing modes for bilingual output:
  - `dual`: Japanese left channel, Chinese right channel
  - `overlay`: Chinese voice overlaid at lower volume
  - `sequential`: Japanese segment followed by Chinese translation
  - `whisper`: Chinese voice at very low volume (subtle)

## Windows One-Click Install (No Technical Background Needed)

Don't want to touch the command line? Use this:

1. From the GitHub page click **Code → Download ZIP**, download and extract
2. Double-click `install.bat` — the script will:
   - Detect/install Python 3.12 (auto-installs if missing, no admin rights needed)
   - Create a virtual environment `.venv`
   - Install Kakure and all dependencies (optional Tsinghua mirror for speed)
   - Auto-download a portable ffmpeg to the project `bin\ffmpeg` directory (no PATH setup needed)
   - Generate the `kakure.toml` config file
3. Double-click `start-kakure.bat` — your browser will open the Kakure UI automatically
4. Enter your OpenAI API Key on the **Settings** page, then upload audio and start

> **Tips**
> - The first run downloads the Whisper model (large-v3 ~3GB), please be patient
> - The one-click install only installs core features (faster-whisper + edge-tts, CPU-friendly)
> - For GPU advanced features (IndexTTS voice cloning, Demucs vocal separation, kotoba-whisper)
>   see the "Manual Install" section below and run `pip install -e ".[optional]"` in the venv

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dependencies
pip install -e ".[dev]"

# For kotoba-whisper ASR backend (optional):
pip install -e ".[kotoba]"

# For IndexTTS TTS backend (optional, requires NVIDIA GPU):
pip install -e ".[indextts]"

# For vocal separation with Demucs (optional, requires GPU for speed):
pip install -e ".[demucs]"
```

Requires [ffmpeg](https://ffmpeg.org/) for audio processing.

## Quick Start

Kakure is used through a local web UI; the CLI only starts the server:

```bash
# Launch Kakure (default: http://127.0.0.1:7860, browser opens automatically)
kakure

# Custom port
kakure --port 8080

# Bind to all interfaces (LAN/remote access)
kakure --host 0.0.0.0

# Do not auto-open a browser window
kakure --no-browser
```

Once the browser opens, use the web UI:

1. Enter your OpenAI API Key on the **Settings** page and configure as needed (ASR/TTS backends, mixing mode, etc.)
2. On the **Process** page, upload your Japanese ASMR audio, click run, and download the bilingual output when done
3. Want transcription only, no translation? Switch to the **Transcribe** page and upload your audio (no API key needed)

## Configuration

Set via environment variables or `.env` file:

```bash
# Required for OpenAI translation
OPENAI_API_KEY=sk-...

# ASR backend: "faster-whisper" or "kotoba-whisper"
KAKURE_ASR_BACKEND=faster-whisper

# TTS backend: "edge-tts" or "indextts"
KAKURE_TTS_BACKEND=edge-tts

# Optional settings
KAKURE_WHISPER_MODEL=large-v3
KAKURE_MIX_MODE=overlay
KAKURE_CHINESE_VOICE=zh-CN-XiaoxiaoNeural

# IndexTTS settings (when KAKURE_TTS_BACKEND=indextts)
KAKURE_INDEXTTS_REFERENCE_AUDIO=path/to/reference_voice.wav
KAKURE_INDEXTTS_MODEL_DIR=  # Leave empty for auto-download

# Vocal separation (Demucs)
KAKURE_SEPARATE_VOCALS=false
KAKURE_DEMUCS_MODEL=htdemucs
KAKURE_VOCALS_VOLUME_DB=-6.0
KAKURE_BACKGROUND_VOLUME_DB=0.0
```

## License

MIT