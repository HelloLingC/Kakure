"""Web UI module - Gradio-based web interface for Kakure."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from kakure.config import (
    ASRBackend,
    ChineseVoice,
    DemucsModel,
    KotobaWhisperModel,
    MixMode,
    Settings,
    TranslationBackend,
    TTSBackend,
    WhisperModelSize,
    load_settings,
    save_settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline flow diagram
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    {"id": "asr", "name": "ASR", "icon": "🎙️", "desc": "Transcribe Japanese"},
    {"id": "translate", "name": "Translation", "icon": "🌐", "desc": "JP → CN"},
    {"id": "tts", "name": "TTS", "icon": "🔊", "desc": "Generate Chinese voice"},
    {"id": "separate", "name": "Vocal Sep.", "icon": "🎵", "desc": "Separate vocals"},
    {"id": "mix", "name": "Mixing", "icon": "🎛️", "desc": "Mix audio tracks"},
    {"id": "export", "name": "Export", "icon": "💾", "desc": "Save output file"},
]

TRANSCRIBE_STAGES = [
    {"id": "load", "name": "Load Model", "icon": "📦", "desc": "Load ASR model"},
    {"id": "transcribe", "name": "Transcribe", "icon": "🎙️", "desc": "Speech recognition"},
]

_PIPELINE_CSS = """<style>
@keyframes kakure-step-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
  50% { box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.1); }
}
.kakure-pipeline {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: 10px 0; font-family: system-ui, -apple-system, sans-serif;
}
.kakure-step {
  display: flex; align-items: center; gap: 8px; padding: 8px 14px;
  border-radius: 10px; border: 2px solid; min-width: 110px;
  transition: all 0.3s ease;
}
.kakure-step-icon { font-size: 22px; line-height: 1; }
.kakure-step-name { font-weight: 600; font-size: 13px; }
.kakure-step-desc { font-size: 11px; }
.kakure-arrow { color: #d1d5db; font-size: 20px; font-weight: 300; }
.kakure-step.pending { border-color: #e5e7eb; background: #f9fafb; }
.kakure-step.pending .kakure-step-name { color: #6b7280; }
.kakure-step.pending .kakure-step-desc { color: #9ca3af; }
.kakure-step.running {
  border-color: #3b82f6; background: #eff6ff;
  animation: kakure-step-pulse 2s ease-in-out infinite;
}
.kakure-step.running .kakure-step-name { color: #1d4ed8; }
.kakure-step.running .kakure-step-desc { color: #3b82f6; }
.kakure-step.completed { border-color: #22c55e; background: #f0fdf4; }
.kakure-step.completed .kakure-step-name { color: #15803d; }
.kakure-step.completed .kakure-step-desc { color: #16a34a; }
.kakure-step.skipped {
  border-color: #e5e7eb; background: #f9fafb; border-style: dashed;
}
.kakure-step.skipped .kakure-step-name { color: #9ca3af; }
.kakure-step.skipped .kakure-step-desc { color: #d1d5db; }
.kakure-step.error { border-color: #ef4444; background: #fef2f2; }
.kakure-step.error .kakure-step-name { color: #dc2626; }
.kakure-step.error .kakure-step-desc { color: #ef4444; }
</style>
"""


def _build_pipeline_html(
    stages: list[dict],
    active: str | None = None,
    completed: set[str] | None = None,
    skipped: set[str] | None = None,
    error: str | None = None,
) -> str:
    """Build an HTML flow diagram showing pipeline stage status.

    Args:
        stages: List of stage dicts with 'id', 'name', 'icon', 'desc' keys.
        active: ID of the currently running stage.
        completed: Set of stage IDs that have completed successfully.
        skipped: Set of stage IDs that are skipped.
        error: ID of the stage that encountered an error.

    Returns:
        HTML string for the pipeline flow diagram.
    """
    completed = completed or set()
    skipped = skipped or set()

    parts = []
    for i, stage in enumerate(stages):
        stage_id = stage["id"]
        if stage_id in completed:
            status = "completed"
            desc = f"{stage['desc']} ✓"
        elif stage_id == error:
            status = "error"
            desc = f"{stage['desc']} ✗"
        elif stage_id == active:
            status = "running"
            desc = f"{stage['desc']}..."
        elif stage_id in skipped:
            status = "skipped"
            desc = f"{stage['desc']} (skipped)"
        else:
            status = "pending"
            desc = stage["desc"]

        step_html = (
            f'<div class="kakure-step {status}">'
            f'<span class="kakure-step-icon">{stage["icon"]}</span>'
            f'<div><div class="kakure-step-name">{stage["name"]}</div>'
            f'<div class="kakure-step-desc">{desc}</div></div>'
            f"</div>"
        )
        parts.append(step_html)

        if i < len(stages) - 1:
            parts.append('<span class="kakure-arrow">›</span>')

    return f'{_PIPELINE_CSS}<div class="kakure-pipeline">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# SRT export helper
# ---------------------------------------------------------------------------


def _write_srt(segments: list, path: Path) -> None:
    """Write translated segments to an SRT subtitle file.

    Args:
        segments: List of TranslatedSegment objects.
        path: Output file path.
    """
    lines = []
    for seg in segments:
        start_h, start_rem = divmod(seg.start, 3600)
        start_m, start_s = divmod(start_rem, 60)
        start_ms = int((seg.start - int(seg.start)) * 1000)

        end_h, end_rem = divmod(seg.end, 3600)
        end_m, end_s = divmod(end_rem, 60)
        end_ms = int((seg.end - int(seg.end)) * 1000)

        lines.append(str(seg.id + 1))
        lines.append(
            f"{int(start_h):02d}:{int(start_m):02d}:{int(start_s):02d},{start_ms:03d}"
            f" --> "
            f"{int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d},{end_ms:03d}"
        )
        lines.append(seg.translated_text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------


def process_audio(
    audio_path: str | None,
    mix_mode: str,
    output_format: str,
    bitrate: str,
    progress=gr.Progress(),  # noqa: B008
):
    """Process audio through the full bilingual pipeline.

    Uses saved settings from kakure.toml, overriding only per-file options.
    Yields pipeline progress updates as a generator.

    Yields:
        Tuple of (pipeline_html, output_audio_path, srt_path, status_message, segments_table).
    """
    if audio_path is None:
        yield (
            _build_pipeline_html(PIPELINE_STAGES),
            None,
            None,
            "Please upload an audio file.",
            [],
        )
        return

    completed: set[str] = set()
    skipped: set[str] = set()
    current_stage = "asr"

    try:
        # Load saved settings, override per-file options
        settings = load_settings()
        settings.mix_mode = MixMode(mix_mode)
        settings.output_format = output_format
        settings.output_bitrate = bitrate

        if not settings.separate_vocals:
            skipped.add("separate")

        input_path = Path(audio_path)

        # Step 1: ASR
        current_stage = "asr"
        progress(0.1, desc="Transcribing Japanese audio...")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, active="asr", completed=completed, skipped=skipped
            ),
            None,
            None,
            "Transcribing Japanese audio...",
            [],
        )

        from kakure.asr import create_asr_processor

        asr = create_asr_processor(settings)
        transcription = asr.transcribe(input_path)

        if not transcription.segments:
            completed.add("asr")
            yield (
                _build_pipeline_html(PIPELINE_STAGES, completed=completed, skipped=skipped),
                None,
                None,
                "No speech segments detected in the audio file.",
                [],
            )
            return

        completed.add("asr")

        # Step 2: Translation
        current_stage = "translate"
        progress(0.3, desc="Translating to Chinese...")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, active="translate", completed=completed, skipped=skipped
            ),
            None,
            None,
            "Translating to Chinese...",
            [],
        )

        from kakure.translator import TranslatedSegment, Translator

        translated = [
            TranslatedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text="",
            )
            for seg in transcription.segments
        ]
        translator = Translator(settings)
        translated = translator.translate_segments(translated)
        completed.add("translate")

        # Build segment dicts for TTS and mixing
        segment_dicts = [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "original_text": seg.original_text,
                "translated_text": seg.translated_text,
            }
            for seg in translated
        ]

        # Step 3: TTS
        current_stage = "tts"
        progress(0.5, desc="Generating Chinese voice...")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, active="tts", completed=completed, skipped=skipped
            ),
            None,
            None,
            "Generating Chinese voice...",
            [],
        )

        from kakure.tts import create_tts_processor

        tts = create_tts_processor(settings)
        tts_results = tts.generate_sync(segment_dicts)
        tts_dicts = [
            {
                "segment_id": r.segment_id,
                "audio_path": str(r.audio_path),
                "duration_ms": r.duration_ms,
            }
            for r in tts_results
        ]
        completed.add("tts")

        # Step 4: Vocal separation (optional)
        separated = None
        if settings.separate_vocals:
            current_stage = "separate"
            progress(0.7, desc="Separating vocals from background...")
            yield (
                _build_pipeline_html(
                    PIPELINE_STAGES, active="separate", completed=completed, skipped=skipped
                ),
                None,
                None,
                "Separating vocals from background...",
                [],
            )

            from kakure.separator import VocalSeparator

            separator = VocalSeparator(settings)
            separated = separator.separate(input_path)
            completed.add("separate")

        # Step 5: Mixing
        current_stage = "mix"
        progress(0.85, desc="Mixing audio tracks...")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, active="mix", completed=completed, skipped=skipped
            ),
            None,
            None,
            "Mixing audio tracks...",
            [],
        )

        from pydub import AudioSegment

        from kakure.mixer import AudioMixer, MixInput

        original_audio = AudioSegment.from_file(str(input_path))
        mix_input = MixInput(
            original_audio=original_audio,
            segments=segment_dicts,
            tts_results=tts_dicts,
            separated=separated,
        )
        mixer = AudioMixer(settings)
        mixed_audio = mixer.mix(mix_input)
        completed.add("mix")

        # Step 6: Export audio
        current_stage = "export"
        progress(0.95, desc="Exporting final audio...")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, active="export", completed=completed, skipped=skipped
            ),
            None,
            None,
            "Exporting final audio...",
            [],
        )

        output_path = input_path.parent / f"{input_path.stem}_bilingual.{output_format}"

        mixer.export(
            mixed_audio,
            output_path,
            format=output_format,
            bitrate=bitrate,
            sample_rate=settings.output_sample_rate,
        )

        # Export SRT subtitle file
        srt_path = input_path.parent / f"{input_path.stem}_bilingual.srt"
        _write_srt(translated, srt_path)

        duration = len(mixed_audio) / 1000.0
        status = (
            f"Done! Duration: {duration:.1f}s | "
            f"Segments: {len(transcription.segments)} | "
            f"Mode: {settings.mix_mode.value}"
        )
        if separated:
            status += " | Vocals separated"

        # Build segments table
        segments_table = [
            [seg.id, f"{seg.start:.2f}", f"{seg.end:.2f}", seg.original_text, seg.translated_text]
            for seg in translated
        ]

        completed.add("export")
        progress(1.0, desc="Complete!")

        yield (
            _build_pipeline_html(PIPELINE_STAGES, completed=completed, skipped=skipped),
            str(output_path),
            str(srt_path),
            status,
            segments_table,
        )

    except Exception as e:
        logger.exception("Processing failed")
        yield (
            _build_pipeline_html(
                PIPELINE_STAGES, completed=completed, skipped=skipped, error=current_stage
            ),
            None,
            None,
            f"Error: {e}",
            [],
        )


def transcribe_audio(
    audio_path: str | None,
    progress=gr.Progress(),  # noqa: B008
):
    """Transcribe audio using ASR only. Uses saved settings from kakure.toml.

    Yields:
        Tuple of (pipeline_html, segments_table, info_text).
    """
    if audio_path is None:
        yield (
            _build_pipeline_html(TRANSCRIBE_STAGES),
            [],
            "Please upload an audio file.",
        )
        return

    completed: set[str] = set()
    current_stage = "load"

    try:
        settings = load_settings()

        # Step 1: Load model
        progress(0.3, desc="Loading ASR model...")
        yield (
            _build_pipeline_html(TRANSCRIBE_STAGES, active="load", completed=completed),
            [],
            "Loading ASR model...",
        )

        from kakure.asr import create_asr_processor

        asr = create_asr_processor(settings)
        completed.add("load")

        # Step 2: Transcribe
        current_stage = "transcribe"
        progress(0.5, desc="Transcribing...")
        yield (
            _build_pipeline_html(TRANSCRIBE_STAGES, active="transcribe", completed=completed),
            [],
            "Transcribing...",
        )

        result = asr.transcribe(Path(audio_path))
        completed.add("transcribe")

        progress(1.0, desc="Complete!")

        rows = [[seg.id, f"{seg.start:.2f}", f"{seg.end:.2f}", seg.text] for seg in result.segments]

        info = (
            f"Language: {result.language} "
            f"({result.language_probability:.1%} confidence) | "
            f"Duration: {result.duration:.1f}s | "
            f"Segments: {len(result.segments)}"
        )

        yield (
            _build_pipeline_html(TRANSCRIBE_STAGES, completed=completed),
            rows,
            info,
        )

    except Exception as e:
        logger.exception("Transcription failed")
        yield (
            _build_pipeline_html(TRANSCRIBE_STAGES, completed=completed, error=current_stage),
            [],
            f"Error: {e}",
        )


# ---------------------------------------------------------------------------
# Settings save/reset helpers
# ---------------------------------------------------------------------------


def save_settings_to_file(
    asr_backend: str,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    whisper_language: str,
    whisper_beam_size: int,
    whisper_vad_filter: bool,
    kotoba_whisper_model: str,
    kotoba_whisper_chunk_length_s: int,
    translation_backend: str,
    openai_api_key: str,
    openai_base_url: str,
    openai_model: str,
    deepl_api_key: str,
    tts_backend: str,
    chinese_voice: str,
    tts_rate: str,
    tts_volume: str,
    tts_pitch: str,
    indextts_reference_audio: str,
    indextts_model_dir: str,
    indextts_language: str,
    mix_mode: str,
    overlay_volume_db: float,
    whisper_volume_db: float,
    spatial_cross_db: float,
    sequential_gap_ms: int,
    separate_vocals: bool,
    demucs_model: str,
    demucs_device: str,
    vocals_volume_db: float,
    background_volume_db: float,
    output_format: str,
    output_bitrate: str,
    output_sample_rate: int,
    temp_dir: str,
) -> str:
    """Save settings from form values to the config file.

    Returns:
        Status message string.
    """
    try:
        settings = Settings(
            asr_backend=ASRBackend(asr_backend),
            whisper_model=WhisperModelSize(whisper_model),
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_language=whisper_language,
            whisper_beam_size=whisper_beam_size,
            whisper_vad_filter=whisper_vad_filter,
            kotoba_whisper_model=KotobaWhisperModel(kotoba_whisper_model),
            kotoba_whisper_chunk_length_s=kotoba_whisper_chunk_length_s,
            translation_backend=TranslationBackend(translation_backend),
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            deepl_api_key=deepl_api_key,
            tts_backend=TTSBackend(tts_backend),
            chinese_voice=ChineseVoice(chinese_voice),
            tts_rate=tts_rate,
            tts_volume=tts_volume,
            tts_pitch=tts_pitch,
            indextts_reference_audio=indextts_reference_audio,
            indextts_model_dir=indextts_model_dir,
            indextts_language=indextts_language,
            mix_mode=MixMode(mix_mode),
            overlay_volume_db=overlay_volume_db,
            whisper_volume_db=whisper_volume_db,
            spatial_cross_db=spatial_cross_db,
            sequential_gap_ms=sequential_gap_ms,
            separate_vocals=separate_vocals,
            demucs_model=DemucsModel(demucs_model),
            demucs_device=demucs_device,
            vocals_volume_db=vocals_volume_db,
            background_volume_db=background_volume_db,
            output_format=output_format,
            output_bitrate=output_bitrate,
            output_sample_rate=output_sample_rate,
            temp_dir=temp_dir,
        )
        save_settings(settings)
        return "✓ Settings saved to kakure.toml"
    except Exception as e:
        logger.exception("Failed to save settings")
        return f"Error saving settings: {e}"


def _settings_to_form_values(settings: Settings) -> tuple:
    """Convert a Settings object to the tuple of form values + visibility updates.

    Used by both load_settings_to_form and reset_settings to avoid duplication.
    """
    is_fw = settings.asr_backend == ASRBackend.FASTER_WHISPER
    is_kw = settings.asr_backend == ASRBackend.KOTOBA_WHISPER
    is_edge = settings.tts_backend == TTSBackend.EDGE_TTS
    is_index = settings.tts_backend == TTSBackend.INDEX_TTS
    return (
        settings.asr_backend.value,
        settings.whisper_model.value,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.whisper_language,
        settings.whisper_beam_size,
        settings.whisper_vad_filter,
        settings.kotoba_whisper_model.value,
        settings.kotoba_whisper_chunk_length_s,
        settings.translation_backend.value,
        settings.openai_api_key,
        settings.openai_base_url,
        settings.openai_model,
        settings.deepl_api_key,
        settings.tts_backend.value,
        settings.chinese_voice.value,
        settings.tts_rate,
        settings.tts_volume,
        settings.tts_pitch,
        settings.indextts_reference_audio,
        settings.indextts_model_dir,
        settings.indextts_language,
        settings.mix_mode.value,
        settings.overlay_volume_db,
        settings.whisper_volume_db,
        settings.spatial_cross_db,
        settings.sequential_gap_ms,
        settings.separate_vocals,
        settings.demucs_model.value,
        settings.demucs_device,
        settings.vocals_volume_db,
        settings.background_volume_db,
        settings.output_format,
        settings.output_bitrate,
        settings.output_sample_rate,
        settings.temp_dir,
        # Visibility updates for ASR fields
        gr.update(visible=is_fw),  # whisper_model
        gr.update(visible=is_fw),  # whisper_compute_type
        gr.update(visible=is_fw),  # whisper_language
        gr.update(visible=is_fw),  # whisper_beam_size
        gr.update(visible=is_fw),  # whisper_vad_filter
        gr.update(visible=is_kw),  # kotoba_whisper_model
        gr.update(visible=is_kw),  # kotoba_whisper_chunk_length_s
        # Visibility updates for TTS fields
        gr.update(visible=is_edge),  # chinese_voice
        gr.update(visible=is_edge),  # tts_rate
        gr.update(visible=is_edge),  # tts_volume
        gr.update(visible=is_edge),  # tts_pitch
        gr.update(visible=is_index),  # indextts_reference_audio
        gr.update(visible=is_index),  # indextts_model_dir
        gr.update(visible=is_index),  # indextts_language
    )


def load_settings_to_form() -> tuple:
    """Load settings from kakure.toml and return all form values + visibility.

    Called on page load to populate the Settings tab with saved values.
    """
    settings = load_settings()
    return _settings_to_form_values(settings)


def load_process_defaults() -> tuple:
    """Load process-tab defaults from kakure.toml.

    Returns:
        Tuple of (mix_mode, output_format, bitrate).
    """
    settings = load_settings()
    return settings.mix_mode.value, settings.output_format, settings.output_bitrate


def reset_settings() -> tuple:
    """Reset settings to defaults, save to config file, and return all form values.

    Returns:
        Tuple of all default form values, visibility updates, and a status message.
    """
    try:
        settings = Settings()
        save_settings(settings)
        return (*_settings_to_form_values(settings), "✓ Settings reset to defaults")
    except Exception as e:
        logger.exception("Failed to reset settings")
        settings = Settings()
        return (*_settings_to_form_values(settings), f"Error resetting settings: {e}")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Create the Gradio web UI application."""

    # Enum choices for dropdowns
    mix_modes = [m.value for m in MixMode]
    asr_backends = [b.value for b in ASRBackend]
    whisper_models = [s.value for s in WhisperModelSize]
    kotoba_models = [m.value for m in KotobaWhisperModel]
    tts_backends = [b.value for b in TTSBackend]
    voice_choices = [(f"{v.value} ({_voice_description(v)})", v.value) for v in ChineseVoice]
    demucs_models = [m.value for m in DemucsModel]
    translation_backends = [b.value for b in TranslationBackend]
    output_formats = ["mp3", "wav", "m4a", "flac", "ogg"]
    bitrates = ["128k", "192k", "256k", "320k"]

    # Load defaults from config file
    defaults = load_settings()

    with gr.Blocks(
        title="Kakure - ASMR Bilingual Voice Overlay",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Kakure\n"
            "ASMR Japanese-to-Chinese bilingual voice overlay tool.  \n"
            "Upload Japanese ASMR audio and overlay a Chinese translation voice track."
        )

        with gr.Tabs():
            # ============================================================
            # Process Tab
            # ============================================================
            with gr.Tab("Process"):
                audio_input = gr.Audio(
                    label="Upload Japanese Audio",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Row():
                    mix_mode = gr.Dropdown(
                        choices=mix_modes,
                        value=defaults.mix_mode.value,
                        label="Mix Mode",
                        info="How to combine Japanese and Chinese audio",
                    )
                    output_format = gr.Dropdown(
                        choices=output_formats,
                        value=defaults.output_format,
                        label="Output Format",
                    )
                    bitrate = gr.Dropdown(
                        choices=bitrates,
                        value=defaults.output_bitrate,
                        label="Bitrate",
                    )

                gr.Markdown(
                    "*All other settings (ASR, TTS, volumes, etc.) are configured in the "
                    "Settings tab and saved to `kakure.toml`.*"
                )

                process_btn = gr.Button("Process", variant="primary", size="lg")

                pipeline_html = gr.HTML(
                    value=_build_pipeline_html(PIPELINE_STAGES),
                )
                process_status = gr.Textbox(label="Status", interactive=False)
                audio_output = gr.Audio(label="Output Audio", type="filepath")
                srt_output = gr.File(label="SRT Subtitles")
                segments_table = gr.Dataframe(
                    headers=["#", "Start", "End", "Japanese", "Chinese"],
                    label="Translated Segments",
                    wrap=True,
                )

            # ============================================================
            # Transcribe Tab
            # ============================================================
            with gr.Tab("Transcribe"):
                transcribe_input = gr.Audio(
                    label="Upload Audio for Transcription",
                    type="filepath",
                    sources=["upload"],
                )

                gr.Markdown("*ASR settings are configured in the Settings tab.*")

                transcribe_btn = gr.Button("Transcribe", variant="primary")

                transcribe_pipeline_html = gr.HTML(
                    value=_build_pipeline_html(TRANSCRIBE_STAGES),
                )
                transcribe_info = gr.Textbox(label="Info", interactive=False)
                transcribe_table = gr.Dataframe(
                    headers=["#", "Start (s)", "End (s)", "Text"],
                    label="Transcription",
                    wrap=True,
                )

            # ============================================================
            # Settings Tab
            # ============================================================
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("ASR Settings", open=True):
                            sett_asr_backend = gr.Dropdown(
                                choices=asr_backends,
                                value=defaults.asr_backend.value,
                                label="ASR Backend",
                            )
                            sett_whisper_model = gr.Dropdown(
                                choices=whisper_models,
                                value=defaults.whisper_model.value,
                                label="Whisper Model",
                                visible=defaults.asr_backend == ASRBackend.FASTER_WHISPER,
                            )
                            sett_whisper_device = gr.Dropdown(
                                choices=["cpu", "cuda"],
                                value=defaults.whisper_device,
                                label="Device",
                            )
                            # faster-whisper specific
                            sett_whisper_compute_type = gr.Textbox(
                                value=defaults.whisper_compute_type,
                                label="Compute Type",
                                info="e.g. int8, float16, int8_float16",
                                visible=defaults.asr_backend == ASRBackend.FASTER_WHISPER,
                            )
                            sett_whisper_language = gr.Textbox(
                                value=defaults.whisper_language,
                                label="Language",
                                info="Source language code (e.g. ja)",
                                visible=defaults.asr_backend == ASRBackend.FASTER_WHISPER,
                            )
                            sett_whisper_beam_size = gr.Number(
                                value=defaults.whisper_beam_size,
                                label="Beam Size",
                                minimum=1,
                                maximum=20,
                                step=1,
                                visible=defaults.asr_backend == ASRBackend.FASTER_WHISPER,
                            )
                            sett_whisper_vad_filter = gr.Checkbox(
                                value=defaults.whisper_vad_filter,
                                label="VAD Filter",
                                info="Filter out non-speech segments",
                                visible=defaults.asr_backend == ASRBackend.FASTER_WHISPER,
                            )
                            # kotoba-whisper specific
                            sett_kotoba_whisper_model = gr.Dropdown(
                                choices=kotoba_models,
                                value=defaults.kotoba_whisper_model.value,
                                label="Kotoba-Whisper Model",
                                visible=defaults.asr_backend == ASRBackend.KOTOBA_WHISPER,
                            )
                            sett_kotoba_whisper_chunk_length_s = gr.Number(
                                value=defaults.kotoba_whisper_chunk_length_s,
                                label="Chunk Length (s)",
                                minimum=5,
                                maximum=60,
                                step=1,
                                visible=defaults.asr_backend == ASRBackend.KOTOBA_WHISPER,
                            )

                        with gr.Accordion("Translation Settings", open=True):
                            sett_translation_backend = gr.Dropdown(
                                choices=translation_backends,
                                value=defaults.translation_backend.value,
                                label="Translation Backend",
                            )
                            sett_openai_api_key = gr.Textbox(
                                value=defaults.openai_api_key,
                                label="OpenAI API Key",
                                type="password",
                                placeholder="sk-...",
                            )
                            sett_openai_base_url = gr.Textbox(
                                value=defaults.openai_base_url,
                                label="OpenAI Base URL",
                                placeholder="https://api.openai.com/v1",
                            )
                            sett_openai_model = gr.Textbox(
                                value=defaults.openai_model,
                                label="OpenAI Model",
                                placeholder="gpt-4o-mini",
                            )
                            sett_deepl_api_key = gr.Textbox(
                                value=defaults.deepl_api_key,
                                label="DeepL API Key",
                                type="password",
                                placeholder="DeepL authentication key",
                            )

                        with gr.Accordion("TTS Settings", open=True):
                            sett_tts_backend = gr.Dropdown(
                                choices=tts_backends,
                                value=defaults.tts_backend.value,
                                label="TTS Backend",
                            )
                            sett_chinese_voice = gr.Dropdown(
                                choices=voice_choices,
                                value=defaults.chinese_voice.value,
                                label="Chinese Voice (edge-tts)",
                                visible=defaults.tts_backend == TTSBackend.EDGE_TTS,
                            )
                            sett_tts_rate = gr.Textbox(
                                value=defaults.tts_rate,
                                label="Speech Rate",
                                placeholder="+0%",
                                visible=defaults.tts_backend == TTSBackend.EDGE_TTS,
                            )
                            sett_tts_volume = gr.Textbox(
                                value=defaults.tts_volume,
                                label="TTS Volume",
                                placeholder="+0%",
                                visible=defaults.tts_backend == TTSBackend.EDGE_TTS,
                            )
                            sett_tts_pitch = gr.Textbox(
                                value=defaults.tts_pitch,
                                label="Pitch",
                                placeholder="+0Hz",
                                visible=defaults.tts_backend == TTSBackend.EDGE_TTS,
                            )
                            sett_indextts_reference_audio = gr.Textbox(
                                value=defaults.indextts_reference_audio,
                                label="IndexTTS Reference Audio",
                                placeholder="Path to reference audio file",
                                visible=defaults.tts_backend == TTSBackend.INDEX_TTS,
                            )
                            sett_indextts_model_dir = gr.Textbox(
                                value=defaults.indextts_model_dir,
                                label="IndexTTS Model Directory",
                                placeholder="Leave empty for auto-download",
                                visible=defaults.tts_backend == TTSBackend.INDEX_TTS,
                            )
                            sett_indextts_language = gr.Textbox(
                                value=defaults.indextts_language,
                                label="IndexTTS Language",
                                placeholder="zh",
                                visible=defaults.tts_backend == TTSBackend.INDEX_TTS,
                            )

                    with gr.Column():
                        with gr.Accordion("Mixing Settings", open=True):
                            sett_mix_mode = gr.Dropdown(
                                choices=mix_modes,
                                value=defaults.mix_mode.value,
                                label="Mix Mode",
                            )
                            sett_overlay_volume_db = gr.Slider(
                                minimum=-30,
                                maximum=0,
                                value=defaults.overlay_volume_db,
                                step=0.5,
                                label="Overlay Volume (dB)",
                            )
                            sett_whisper_volume_db = gr.Slider(
                                minimum=-30,
                                maximum=0,
                                value=defaults.whisper_volume_db,
                                step=0.5,
                                label="Whisper Volume (dB)",
                            )
                            sett_spatial_cross_db = gr.Slider(
                                minimum=-30,
                                maximum=0,
                                value=defaults.spatial_cross_db,
                                step=0.5,
                                label="Spatial Cross Volume (dB)",
                            )
                            sett_sequential_gap_ms = gr.Number(
                                value=defaults.sequential_gap_ms,
                                label="Sequential Gap (ms)",
                                minimum=0,
                                maximum=5000,
                                step=100,
                            )

                        with gr.Accordion("Vocal Separation", open=True):
                            sett_separate_vocals = gr.Checkbox(
                                value=defaults.separate_vocals,
                                label="Separate Vocals",
                                info="Use Demucs to separate vocals from background",
                            )
                            sett_demucs_model = gr.Dropdown(
                                choices=demucs_models,
                                value=defaults.demucs_model.value,
                                label="Demucs Model",
                            )
                            sett_demucs_device = gr.Dropdown(
                                choices=["cpu", "cuda"],
                                value=defaults.demucs_device,
                                label="Demucs Device",
                            )
                            sett_vocals_volume_db = gr.Slider(
                                minimum=-30,
                                maximum=0,
                                value=defaults.vocals_volume_db,
                                step=0.5,
                                label="Vocals Volume (dB)",
                            )
                            sett_background_volume_db = gr.Slider(
                                minimum=-30,
                                maximum=10,
                                value=defaults.background_volume_db,
                                step=0.5,
                                label="Background Volume (dB)",
                            )

                        with gr.Accordion("Output Settings", open=True):
                            sett_output_format = gr.Dropdown(
                                choices=output_formats,
                                value=defaults.output_format,
                                label="Output Format",
                            )
                            sett_output_bitrate = gr.Dropdown(
                                choices=bitrates,
                                value=defaults.output_bitrate,
                                label="Output Bitrate",
                            )
                            sett_output_sample_rate = gr.Number(
                                value=defaults.output_sample_rate,
                                label="Sample Rate",
                                minimum=8000,
                                maximum=192000,
                                step=1000,
                            )

                        with gr.Accordion("Paths", open=True):
                            sett_temp_dir = gr.Textbox(
                                value=defaults.temp_dir,
                                label="Temp Directory",
                                placeholder="/tmp/kakure",
                            )

                with gr.Row():
                    save_btn = gr.Button("Save Settings", variant="primary", size="lg")
                    reset_btn = gr.Button("Reset to Defaults", size="lg")

                settings_status = gr.Textbox(label="Settings Status", interactive=False)

        # ================================================================
        # Event handlers
        # ================================================================

        # Settings tab: conditional visibility for ASR backend
        def _toggle_asr_fields(asr_backend: str):
            is_fw = asr_backend == "faster-whisper"
            is_kw = asr_backend == "kotoba-whisper"
            return (
                gr.update(visible=is_fw),  # whisper_model
                gr.update(visible=is_fw),  # whisper_compute_type
                gr.update(visible=is_fw),  # whisper_language
                gr.update(visible=is_fw),  # whisper_beam_size
                gr.update(visible=is_fw),  # whisper_vad_filter
                gr.update(visible=is_kw),  # kotoba_whisper_model
                gr.update(visible=is_kw),  # kotoba_whisper_chunk_length_s
            )

        sett_asr_backend.change(
            fn=_toggle_asr_fields,
            inputs=[sett_asr_backend],
            outputs=[
                sett_whisper_model,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
            ],
        )

        # Settings tab: conditional visibility for TTS backend
        def _toggle_tts_fields(tts_backend: str):
            is_edge = tts_backend == "edge-tts"
            is_index = tts_backend == "indextts"
            return (
                gr.update(visible=is_edge),  # chinese_voice
                gr.update(visible=is_edge),  # tts_rate
                gr.update(visible=is_edge),  # tts_volume
                gr.update(visible=is_edge),  # tts_pitch
                gr.update(visible=is_index),  # indextts_reference_audio
                gr.update(visible=is_index),  # indextts_model_dir
                gr.update(visible=is_index),  # indextts_language
            )

        sett_tts_backend.change(
            fn=_toggle_tts_fields,
            inputs=[sett_tts_backend],
            outputs=[
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
            ],
        )

        # Process tab: run pipeline
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, mix_mode, output_format, bitrate],
            outputs=[pipeline_html, audio_output, srt_output, process_status, segments_table],
        )

        # Transcribe tab: run ASR
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[transcribe_input],
            outputs=[transcribe_pipeline_html, transcribe_table, transcribe_info],
        )

        # Settings tab: save button
        save_btn.click(
            fn=save_settings_to_file,
            inputs=[
                sett_asr_backend,
                sett_whisper_model,
                sett_whisper_device,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
                sett_translation_backend,
                sett_openai_api_key,
                sett_openai_base_url,
                sett_openai_model,
                sett_deepl_api_key,
                sett_tts_backend,
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
                sett_mix_mode,
                sett_overlay_volume_db,
                sett_whisper_volume_db,
                sett_spatial_cross_db,
                sett_sequential_gap_ms,
                sett_separate_vocals,
                sett_demucs_model,
                sett_demucs_device,
                sett_vocals_volume_db,
                sett_background_volume_db,
                sett_output_format,
                sett_output_bitrate,
                sett_output_sample_rate,
                sett_temp_dir,
            ],
            outputs=[settings_status],
        )

        # Settings tab: reset button
        reset_btn.click(
            fn=reset_settings,
            inputs=[],
            outputs=[
                sett_asr_backend,
                sett_whisper_model,
                sett_whisper_device,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
                sett_translation_backend,
                sett_openai_api_key,
                sett_openai_base_url,
                sett_openai_model,
                sett_deepl_api_key,
                sett_tts_backend,
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
                sett_mix_mode,
                sett_overlay_volume_db,
                sett_whisper_volume_db,
                sett_spatial_cross_db,
                sett_sequential_gap_ms,
                sett_separate_vocals,
                sett_demucs_model,
                sett_demucs_device,
                sett_vocals_volume_db,
                sett_background_volume_db,
                sett_output_format,
                sett_output_bitrate,
                sett_output_sample_rate,
                sett_temp_dir,
                # Visibility updates for ASR fields
                sett_whisper_model,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
                # Visibility updates for TTS fields
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
                settings_status,
            ],
        )

        # Load saved settings on page refresh — Process tab
        app.load(
            fn=load_process_defaults,
            inputs=[],
            outputs=[mix_mode, output_format, bitrate],
        )

        # Load saved settings on page refresh — Settings tab
        app.load(
            fn=load_settings_to_form,
            inputs=[],
            outputs=[
                sett_asr_backend,
                sett_whisper_model,
                sett_whisper_device,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
                sett_translation_backend,
                sett_openai_api_key,
                sett_openai_base_url,
                sett_openai_model,
                sett_deepl_api_key,
                sett_tts_backend,
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
                sett_mix_mode,
                sett_overlay_volume_db,
                sett_whisper_volume_db,
                sett_spatial_cross_db,
                sett_sequential_gap_ms,
                sett_separate_vocals,
                sett_demucs_model,
                sett_demucs_device,
                sett_vocals_volume_db,
                sett_background_volume_db,
                sett_output_format,
                sett_output_bitrate,
                sett_output_sample_rate,
                sett_temp_dir,
                # Visibility updates for ASR fields
                sett_whisper_model,
                sett_whisper_compute_type,
                sett_whisper_language,
                sett_whisper_beam_size,
                sett_whisper_vad_filter,
                sett_kotoba_whisper_model,
                sett_kotoba_whisper_chunk_length_s,
                # Visibility updates for TTS fields
                sett_chinese_voice,
                sett_tts_rate,
                sett_tts_volume,
                sett_tts_pitch,
                sett_indextts_reference_audio,
                sett_indextts_model_dir,
                sett_indextts_language,
            ],
        )

    return app


def _voice_description(voice: ChineseVoice) -> str:
    """Return a short human-readable description for a Chinese voice."""
    descriptions = {
        ChineseVoice.XIAOXIAO: "Female, warm",
        ChineseVoice.XIAOYI: "Female, gentle",
        ChineseVoice.YUNJIAN: "Male, calm",
        ChineseVoice.YUNXI: "Male, warm",
        ChineseVoice.YUNXIA: "Female, sweet",
        ChineseVoice.YUNYANG: "Male, news anchor",
    }
    return descriptions.get(voice, "")


def launch_webui(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> None:
    """Launch the Gradio web UI.

    Args:
        share: Whether to create a public share link.
        server_name: Server hostname to bind to.
        server_port: Server port to listen on.
    """
    app = create_app()
    app.launch(share=share, server_name=server_name, server_port=server_port)


def main() -> None:
    """CLI entry point — launch the Kakure web UI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kakure",
        description="Kakure - ASMR Japanese-to-Chinese bilingual voice overlay tool",
    )
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument(
        "--server-name", default="0.0.0.0", help="Server hostname (default: 0.0.0.0)"
    )
    parser.add_argument("--server-port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    launch_webui(share=args.share, server_name=args.server_name, server_port=args.server_port)
