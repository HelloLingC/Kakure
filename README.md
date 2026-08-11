# Kakure (隠れ)

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

```bash
# Process an ASMR audio file (default: faster-whisper ASR, edge-tts TTS)
kakure process input.mp3

# Specify output and mixing mode
kakure process input.mp3 -o output.mp3 -m dual

# Full pipeline with kotoba-whisper ASR
kakure process input.mp3 --asr-backend kotoba-whisper -o output.mp3

# Full pipeline with IndexTTS (voice cloning from reference audio)
kakure process input.mp3 --tts-backend indextts --reference-audio voice_sample.wav -o output.mp3

# Process with vocal separation for cleaner mixing
kakure process input.mp3 --separate-vocals -o output.mp3

# Transcribe only (no translation)
kakure transcribe input.mp3

# ASR only with kotoba-whisper
kakure transcribe input.mp3 --asr-backend kotoba-whisper

# List available Chinese voices (edge-tts)
kakure voices

# Show current configuration
kakure config
```

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