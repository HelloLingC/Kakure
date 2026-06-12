"""Configuration module for Kakure."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class MixMode(str, Enum):
    """Audio mixing mode for bilingual output."""

    DUAL = "dual"  # Japanese left channel, Chinese right channel
    OVERLAY = "overlay"  # Chinese voice overlaid at lower volume
    SEQUENTIAL = "sequential"  # Japanese segment then Chinese translation
    WHISPER = "whisper"  # Chinese voice at very low volume (subtle)


class ASRBackend(str, Enum):
    """ASR engine backend."""

    FASTER_WHISPER = "faster-whisper"
    KOTOBA_WHISPER = "kotoba-whisper"


class TranslationBackend(str, Enum):
    """Translation service backend."""

    OPENAI = "openai"
    DEEPL = "deepl"


class WhisperModelSize(str, Enum):
    """faster-whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"
    DISTIL_LARGE_V3 = "distil-large-v3"


class KotobaWhisperModel(str, Enum):
    """kotoba-whisper model variants on HuggingFace."""

    V2_0 = "kotoba-tech/kotoba-whisper-v2.0"
    V2_1 = "kotoba-tech/kotoba-whisper-v2.1"  # + punctuation
    V2_2 = "kotoba-tech/kotoba-whisper-v2.2"  # + speaker diarization


class TTSBackend(str, Enum):
    """TTS engine backend."""

    EDGE_TTS = "edge-tts"
    INDEX_TTS = "indextts"


class ChineseVoice(str, Enum):
    """Available Chinese TTS voices (edge-tts)."""

    XIAOXIAO = "zh-CN-XiaoxiaoNeural"  # Female, warm
    XIAOYI = "zh-CN-XiaoyiNeural"  # Female, gentle
    YUNJIAN = "zh-CN-YunjianNeural"  # Male, calm
    YUNXI = "zh-CN-YunxiNeural"  # Male, warm
    YUNXIA = "zh-CN-YunxiaNeural"  # Female, sweet
    YUNYANG = "zh-CN-YunyangNeural"  # Male, news anchor style


class DemucsModel(str, Enum):
    """Demucs source separation model variants."""

    HTDEMUCS = "htdemucs"  # Default, balanced quality
    HTDEMUCS_FT = "htdemucs_ft"  # Best quality, 4 specialist models
    HTDEMUCS_6S = "htdemucs_6s"  # 6 stems (adds guitar + piano)
    MDX_EXTRA = "mdx_extra"  # Extra training data


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    # ASR backend selection
    asr_backend: ASRBackend = ASRBackend.FASTER_WHISPER

    # faster-whisper settings
    whisper_model: WhisperModelSize = WhisperModelSize.LARGE_V3
    whisper_device: str = Field(default="cpu", description="Device: 'cpu' or 'cuda'")
    whisper_compute_type: str = Field(
        default="int8", description="Compute type: 'float16', 'int8', 'int8_float16'"
    )
    whisper_language: str = "ja"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True

    # kotoba-whisper settings
    kotoba_whisper_model: KotobaWhisperModel = KotobaWhisperModel.V2_0
    kotoba_whisper_chunk_length_s: int = 15

    # Translation
    translation_backend: TranslationBackend = TranslationBackend.OPENAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    deepl_api_key: str = Field(default="", alias="DEEPL_API_KEY")

    # TTS backend selection
    tts_backend: TTSBackend = TTSBackend.EDGE_TTS

    # edge-tts settings
    chinese_voice: ChineseVoice = ChineseVoice.XIAOXIAO
    tts_rate: str = "+0%"  # Speech rate adjustment
    tts_volume: str = "+0%"  # Volume adjustment
    tts_pitch: str = "+0Hz"  # Pitch adjustment

    # IndexTTS settings
    indextts_reference_audio: Path | None = Field(
        default=None,
        description="Path to reference audio for voice cloning (required for IndexTTS). "
        "Should be 3-10 seconds of clean speech.",
    )
    indextts_model_dir: Path | None = Field(
        default=None,
        description="Path to IndexTTS model checkpoints directory. "
        "If None, auto-downloads from HuggingFace on first use.",
    )
    indextts_language: str = Field(
        default="zh",
        description="Language for IndexTTS synthesis.",
    )

    # Mixing
    mix_mode: MixMode = MixMode.OVERLAY
    overlay_volume_db: float = -6.0  # Chinese voice volume relative to original (overlay mode)
    whisper_volume_db: float = -15.0  # Chinese voice volume (whisper mode)
    sequential_gap_ms: int = 800  # Gap between JP and CN in sequential mode

    # Vocal separation (Demucs)
    separate_vocals: bool = False  # Enable vocal/background separation before mixing
    demucs_model: DemucsModel = DemucsModel.HTDEMUCS  # Demucs model variant
    demucs_device: str = Field(
        default="cpu", description="Device for Demucs: 'cpu' or 'cuda'"
    )
    vocals_volume_db: float = -6.0  # Volume for original vocals when separated (dB)
    background_volume_db: float = 0.0  # Volume for background when separated (dB)

    # Output
    output_format: str = "mp3"
    output_bitrate: str = "192k"
    output_sample_rate: int = 44100

    # Paths
    temp_dir: Path = Field(default=Path("/tmp/kakure"))

    model_config = {
        "env_prefix": "KAKURE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings, creating temp dir if needed."""
    settings = Settings()
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return settings