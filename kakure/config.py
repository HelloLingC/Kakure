"""Configuration module for Kakure.

Settings are loaded from and saved to a TOML config file (kakure.toml by default).
If the config file doesn't exist, built-in defaults are used.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import tomlkit
from pydantic import BaseModel, Field

# Default config file path (current working directory)
CONFIG_PATH = Path("kakure.toml")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MixMode(str, Enum):
    """Audio mixing mode for bilingual output."""

    DUAL = "dual"  # Japanese left channel, Chinese right channel
    OVERLAY = "overlay"  # Chinese voice overlaid at lower volume
    SEQUENTIAL = "sequential"  # Japanese segment then Chinese translation
    WHISPER = "whisper"  # Chinese voice at very low volume (subtle)
    SPATIAL = "spatial"  # Cross-panned stereo: JP stronger left, CN stronger right


class ASRBackend(str, Enum):
    """ASR engine backend."""

    FASTER_WHISPER = "faster-whisper"
    KOTOBA_WHISPER = "kotoba-whisper"


class TranslationBackend(str, Enum):
    """Translation service backend."""

    OPENAI = "openai"


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


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Application settings with built-in defaults.

    Load from TOML with ``load_settings()`` or create with defaults via ``Settings()``.
    Save to TOML with ``save_settings()``.
    """

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
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com"
    openai_model: str = "deepseek-v4-flash"
    translation_prompt: str = ""  # Custom system prompt; empty = built-in ASMR prompt
    # Batch translation: segments per API call. 1 = single-segment mode (legacy)
    translation_batch_size: int = 10
    # Approx max input tokens per batch (chars * 1.2 estimate). A batch is
    # split early when this is exceeded, regardless of translation_batch_size.
    translation_batch_token_limit: int = 8000
    # Max concurrent translation API calls. 1 = sequential (legacy). Higher
    # values parallelize batches for faster translation but may hit rate limits.
    translation_max_concurrency: int = Field(default=4, ge=1)

    # TTS backend selection
    tts_backend: TTSBackend = TTSBackend.EDGE_TTS

    # edge-tts settings
    chinese_voice: ChineseVoice = ChineseVoice.XIAOXIAO
    tts_rate: str = "+0%"  # Speech rate adjustment
    tts_volume: str = "+0%"  # Volume adjustment
    tts_pitch: str = "+0Hz"  # Pitch adjustment

    # IndexTTS settings
    indextts_reference_audio: str = ""
    indextts_model_dir: str = ""
    indextts_language: str = "zh"

    # Mixing
    mix_mode: MixMode = MixMode.OVERLAY
    overlay_volume_db: float = -6.0  # Chinese voice volume relative to original (overlay mode)
    whisper_volume_db: float = -15.0  # Chinese voice volume (whisper mode)
    spatial_cross_db: float = -10.5  # Cross-channel volume in spatial mode (~30%, dB)
    sequential_gap_ms: int = 800  # Gap between JP and CN in sequential mode

    # Vocal separation (Demucs)
    separate_vocals: bool = False  # Enable vocal/background separation before mixing
    demucs_model: DemucsModel = DemucsModel.HTDEMUCS  # Demucs model variant
    demucs_device: str = Field(default="cpu", description="Device for Demucs: 'cpu' or 'cuda'")
    vocals_volume_db: float = -6.0  # Volume for original vocals when separated (dB)
    background_volume_db: float = 0.0  # Volume for background when separated (dB)

    # Output
    output_format: str = "mp3"
    output_bitrate: str = "192k"
    output_sample_rate: int = 44100

    # Paths
    # Relative to the running workspace (current working directory).
    temp_dir: str = "tmp"
    # Directory for generated SRT subtitle files. Empty = write in the same
    # directory as the input file (temp_dir in the web UI, input file dir in
    # the CLI).
    srt_output_dir: str = ""

    # Checkpoints
    # Cache pipeline stage results (ASR, translation, TTS, vocal separation)
    # on disk so re-running the same input file skips completed stages.
    enable_checkpoints: bool = True
    # Directory for cached stage results. Empty = <temp_dir>/checkpoints.
    checkpoint_dir: str = ""

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# TOML load / save
# ---------------------------------------------------------------------------


def _settings_to_dict(settings: Settings) -> dict:
    """Convert Settings to a plain dict suitable for TOML serialization.

    Enums are converted to their string values.
    Empty strings for optional path fields are kept as-is.
    """
    data = {}
    for name in settings.model_fields:
        value = getattr(settings, name)
        if isinstance(value, Enum):
            value = value.value
        data[name] = value
    return data


def _dict_to_settings(data: dict) -> Settings:
    """Create Settings from a plain dict (e.g. loaded from TOML).

    Handles string-to-enum and string-to-Path coercion via Pydantic.
    """
    # Filter out keys that aren't in Settings fields
    valid_keys = set(Settings.model_fields)
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Settings(**filtered)


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from a TOML config file.

    If the file doesn't exist, returns Settings with built-in defaults.
    Missing keys in the file fall back to defaults.
    """
    path = path or CONFIG_PATH
    if not path.exists():
        return Settings()
    try:
        with open(path, encoding="utf-8") as f:
            data = tomlkit.load(f)
        return _dict_to_settings(data)
    except Exception:
        # If config file is malformed, fall back to defaults
        return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Save settings to a TOML config file.

    Creates the file if it doesn't exist. Preserves comments and formatting
    in existing files via tomlkit.
    """
    path = path or CONFIG_PATH

    # Load existing document to preserve comments, or create new
    if path.exists():
        with open(path, encoding="utf-8") as f:
            doc = tomlkit.load(f)
    else:
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Kakure Configuration"))
        doc.add(tomlkit.comment("See kakure.toml.example for documentation."))

    # Update values
    data = _settings_to_dict(settings)
    for key, value in data.items():
        doc[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        tomlkit.dump(doc, f)


def get_settings(path: Path | None = None) -> Settings:
    """Load settings from config file and ensure temp dir exists.

    This is the primary entry point for getting settings in application code.
    """
    settings = load_settings(path)
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
    return settings
