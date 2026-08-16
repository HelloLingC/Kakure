"""TTS module - Chinese text-to-speech via the audio.cpp server (audiocpp).

Kakure drives speech synthesis through ``audiocpp_server.exe`` (the audio.cpp
Windows CUDA build bundled under ``./audiocpp``). The server exposes an
OpenAI-compatible REST API; Kakure launches it as a sidecar process, pins the
TTS model via a generated ``server.json``, and calls ``POST /v1/audio/speech``
per translated segment. Voice cloning is configured once through the server's
``default_voice_preset`` (``voice_ref`` + ``reference_text``), so each request
only needs to carry the input text.

The public surface (``BaseTTSProcessor``, ``TTSResult``, ``create_tts_processor``)
is unchanged from the previous backends, so the rest of the pipeline (checkpoint
resume, mixing) keeps working without modification.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydub import AudioSegment

from kakure.config import Settings, TTSBackend

logger = logging.getLogger(__name__)

# HTTP request timeout for a single synthesis call. The first call also triggers
# lazy model loading on the server side, so this must be generous.
_REQUEST_TIMEOUT = 600.0
# How long to wait for the sidecar server to come up after launch.
_STARTUP_TIMEOUT = 120.0


@dataclass
class TTSResult:
    """Result of TTS generation for a single segment."""

    segment_id: int
    audio_path: Path
    duration_ms: int  # Duration in milliseconds


class BaseTTSProcessor(ABC):
    """Abstract base class for TTS processors."""

    @abstractmethod
    def generate_sync(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Generate TTS audio for all translated segments.

        Args:
            segments: List of dicts with 'id', 'translated_text', 'start', 'end'.
            output_dir: Directory to save audio files.

        Returns:
            List of TTSResult with audio paths and durations.
        """
        ...


# ---------------------------------------------------------------------------
# Model catalog: which HuggingFace GGUF to download / auto-discover per family.
# ``file`` is the path inside the ``audio-cpp/audio.cpp-gguf`` snapshot.
# ---------------------------------------------------------------------------

AUDIOCPP_FAMILY_DOWNLOADS: dict[str, dict] = {
    "qwen3_tts": {
        "repo_id": "audio-cpp/audio.cpp-gguf",
        "file": "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
        "label": "Qwen3-TTS 1.7B (q8_0, 中文/日英)",
    },
}

# Families the UI offers. Families not in AUDIOCPP_FAMILY_DOWNLOADS still work
# as long as the user points ``audiocpp_model`` at a local GGUF.
AUDIOCPP_FAMILY_CHOICES: list[str] = [
    "qwen3_tts",
    "index_tts2",
    "glm_tts",
    "outetts",
    "vibevoice",
]


# ---------------------------------------------------------------------------
# Family packages (variants) from audiocpp/model_specs/*.json
#
# Each spec declares a ``family`` (e.g. ``index_tts2``) and a list of
# ``packages`` (variants such as ``index_tts2_5_q8_0``). Kakure exposes those
# variants in the Settings UI so users can pick, say, the IndexTTS2.5 Q8_0
# build without hand-typing a GGUF path. Only single-file GGUF packages are
# surfaced — they map cleanly to ``audiocpp_model``. The ``model_path`` points
# at the default download location used by ``audiocpp/tools/model_manager_v2.py``
# (``audiocpp/models/<target_directory>/<file>`` relative to the project root).
# ---------------------------------------------------------------------------


def _session_options_for(settings: Settings) -> dict[str, str] | None:
    """Emit per-family ``session_options`` for the audiocpp server.

    ``ggml_init`` commits its graph arena to the Windows commit limit up front
    (even though tensors live on the GPU), so the large default arenas used by
    some families (e.g. IndexTTS2.5's 2 GB stages) can fail on low-RAM boxes.
    Shrink those arenas — they only hold tensor metadata and graph descriptors —
    and enable ``mem_saver`` so each stage's graphs are freed after use.
    """
    if settings.audiocpp_family != "index_tts2":
        return None
    return {
        "index_tts2.gpt_graph_arena_mb": "512",
        "index_tts2.s2mel_graph_arena_mb": "512",
        "index_tts2.reference_graph_arena_mb": "256",
        "index_tts2.emotion_text_prefill_graph_arena_mb": "512",
        "index_tts2.emotion_text_decode_graph_arena_mb": "256",
        "index_tts2.weight_context_mb": "32",
        "index_tts2.mem_saver": "true",
    }


def _audiocpp_specs_dir() -> Path:
    """Locate the ``audiocpp/model_specs`` directory."""
    bundled = Path("audiocpp/model_specs")
    if bundled.is_dir():
        return bundled
    repo_root = Path(__file__).resolve().parent.parent / "audiocpp" / "model_specs"
    if repo_root.is_dir():
        return repo_root
    return bundled


def audiocpp_family_packages() -> dict[str, list[dict]]:
    """Return ``{family: [package_info, ...]}`` from ``audiocpp/model_specs``.

    Only GGUF packages are included (single-file models that map cleanly to
    ``audiocpp_model``). Each package info carries the resolved
    ``model_path`` (relative to the project root, pointing at the default
    ``audiocpp/models/<target_directory>/<file>`` location used by
    ``model_manager_v2.py``) and an ``installed`` flag based on whether that
    file currently exists.
    """
    specs_dir = _audiocpp_specs_dir()
    if not specs_dir.is_dir():
        return {}
    result: dict[str, list[dict]] = {}
    for spec_path in sorted(specs_dir.glob("*.json")):
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping model spec %s: %s", spec_path, exc)
            continue
        family = spec.get("family")
        if family not in AUDIOCPP_FAMILY_CHOICES:
            continue
        # Only TTS specs (some specs cover ASR/separation/etc.).
        if spec.get("category") and spec["category"] != "tts":
            continue
        packages: list[dict] = []
        for pkg in spec.get("packages", []):
            if pkg.get("format") != "gguf":
                continue
            files = pkg.get("files") or []
            if not files:
                continue
            remote_file = files[0]
            # Local file name = remote path with ``strip_prefix`` removed.
            strip_prefix = pkg.get("strip_prefix", "")
            local_file = remote_file
            if strip_prefix:
                sp = strip_prefix.rstrip("/")
                if remote_file.startswith(sp + "/"):
                    local_file = remote_file[len(sp) + 1 :]
            target_dir = pkg.get("target_directory", "")
            rel = (
                f"audiocpp/models/{target_dir}/{local_file}"
                if target_dir
                else f"audiocpp/models/{local_file}"
            )
            packages.append(
                {
                    "id": pkg.get("id", ""),
                    "display_name": pkg.get("display_name", pkg.get("id", "")),
                    "precision": pkg.get("precision", ""),
                    "model_path": rel,
                    "installed": Path(rel).exists(),
                    "default": bool(pkg.get("default", False)),
                }
            )
        if packages:
            result[family] = packages
    return result


# ---------------------------------------------------------------------------
# Sidecar server lifecycle
# ---------------------------------------------------------------------------


def _resolve_exe(settings: Settings) -> Path:
    """Locate the audiocpp_server executable."""
    if settings.audiocpp_exe:
        p = Path(settings.audiocpp_exe).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"audiocpp_server.exe not found at: {p}")
        return p
    # 1) bundled next to the project root (where kakure is launched)
    bundled = Path("audiocpp/audiocpp_server.exe")
    if bundled.exists():
        return bundled.resolve()
    # 2) next to this package's parent (repo root) as a fallback
    repo_root = Path(__file__).resolve().parent.parent / "audiocpp" / "audiocpp_server.exe"
    if repo_root.exists():
        return repo_root
    raise FileNotFoundError(
        "audiocpp_server.exe not found. Set audiocpp_exe in kakure.toml to its path, "
        "or place it at ./audiocpp/audiocpp_server.exe."
    )


def _resolve_model_path(settings: Settings) -> Path | None:
    """Resolve the TTS model path.

    Returns ``None`` when no model is configured and none can be auto-discovered
    (the server can still start and use its built-in default voice, but cloning
    will be unavailable).
    """
    if settings.audiocpp_model:
        p = Path(settings.audiocpp_model).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"audiocpp model not found at: {p}")
        return p.resolve()
    info = AUDIOCPP_FAMILY_DOWNLOADS.get(settings.audiocpp_family)
    if not info:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        from kakure.models import hf_cache_dir

        cached = try_to_load_from_cache(
            info["repo_id"], info["file"], cache_dir=str(hf_cache_dir())
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to probe HF cache for auto-discovery: %s", exc)
        return None
    if cached and Path(cached).exists():
        return Path(cached)
    return None


def _hf_cache_dir() -> Path:
    from kakure.models import hf_cache_dir

    return hf_cache_dir()


class _AudioCppServer:
    """Manages the audiocpp_server.exe sidecar process (module-level singleton)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._signature: tuple | None = None
        self._base_url: str | None = None
        self._server_json: Path | None = None
        self._registered_atexit = False

    # -- signature ---------------------------------------------------------
    def _compute_signature(self, settings: Settings, model_path: Path | None) -> tuple:
        ref = str(Path(settings.audiocpp_reference_audio).expanduser().resolve()) \
            if settings.audiocpp_reference_audio else ""
        return (
            settings.audiocpp_host,
            settings.audiocpp_port,
            settings.audiocpp_backend,
            settings.audiocpp_device,
            settings.audiocpp_threads,
            settings.audiocpp_family,
            str(model_path) if model_path else "",
            ref,
            settings.audiocpp_reference_text,
        )

    # -- health ------------------------------------------------------------
    def _health_ok(self, base_url: str) -> bool:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    # -- server.json -------------------------------------------------------
    def _write_server_json(self, settings: Settings, model_path: Path | None, ref_abs: str) -> Path:
        model_entry: dict = {
            "id": "kakure-tts",
            "family": settings.audiocpp_family,
            "task": "tts",
            "mode": "offline",
        }
        if model_path is not None:
            model_entry["path"] = str(model_path)
        if ref_abs:
            preset: dict = {"voice_ref": ref_abs}
            if settings.audiocpp_reference_text:
                preset["reference_text"] = settings.audiocpp_reference_text
            model_entry["default_voice_preset"] = preset

        session_options = _session_options_for(settings)
        if session_options:
            model_entry["session_options"] = session_options

        cfg = {
            "host": settings.audiocpp_host,
            "port": settings.audiocpp_port,
            "backend": settings.audiocpp_backend,
            "device": settings.audiocpp_device,
            "threads": settings.audiocpp_threads,
            "lazy_load": True,
            "models": [model_entry],
        }
        out_dir = Path(settings.temp_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audiocpp_server.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # -- lifecycle ---------------------------------------------------------
    def ensure(self, settings: Settings) -> str:
        """Ensure the sidecar is running for the given settings.

        Returns the base URL of the server. Restarts the process if the
        configuration (signature) changed since it was last launched.
        """
        model_path = _resolve_model_path(settings)
        ref_abs = ""
        if settings.audiocpp_reference_audio:
            rp = Path(settings.audiocpp_reference_audio).expanduser()
            if not rp.is_absolute():
                # resolve relative to cwd (project root) and references/ folder
                for cand in (rp, Path("references") / rp.name):
                    if cand.exists():
                        rp = cand
                        break
            if not rp.exists():
                raise FileNotFoundError(
                    f"audiocpp reference audio not found: {settings.audiocpp_reference_audio}"
                )
            ref_abs = str(rp.resolve())

        signature = self._compute_signature(settings, model_path)
        base_url = f"http://{settings.audiocpp_host}:{settings.audiocpp_port}"

        with self._lock:
            # Already running and matching config?
            if (
                self._proc is not None
                and self._proc.poll() is None
                and self._signature == signature
                and self._health_ok(base_url)
            ):
                return base_url

            # Stale or dead -> restart.
            self._stop_locked()

            exe = _resolve_exe(settings)
            self._server_json = self._write_server_json(settings, model_path, ref_abs)

            log_path = Path(settings.temp_dir).resolve() / "audiocpp_server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Starting audiocpp server: %s --config %s", exe, self._server_json)
            with open(log_path, "ab") as logf:
                self._proc = subprocess.Popen(
                    [str(exe), "--config", str(self._server_json), "--no-ui"],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            self._signature = signature
            self._base_url = base_url
            self._register_atexit()

        # Wait for readiness (poll /health).
        deadline = time.time() + _STARTUP_TIMEOUT
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._dump_log(log_path)
                raise RuntimeError(
                    "audiocpp_server.exe exited during startup (see log above). "
                    "Likely missing CUDA runtime DLLs or a bad model path."
                )
            if self._health_ok(base_url):
                logger.info("audiocpp server is ready at %s", base_url)
                return base_url
            time.sleep(0.5)

        self._dump_log(log_path)
        raise TimeoutError(
            f"audiocpp server did not become ready within {_STARTUP_TIMEOUT}s at {base_url}."
        )

    def _stop_locked(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._signature = None
        self._base_url = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    @staticmethod
    def _dump_log(log_path: Path) -> None:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            if tail.strip():
                logger.error("--- audiocpp_server.log (tail) ---\n%s", tail)
        except Exception:
            pass

    def _register_atexit(self) -> None:
        if not self._registered_atexit:
            atexit.register(self.stop)
            self._registered_atexit = True


_SERVER = _AudioCppServer()


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class AudioCppTTSProcessor(BaseTTSProcessor):
    """Chinese speech synthesis through the audio.cpp server (audiocpp).

    Launches/reattaches ``audiocpp_server.exe`` as a sidecar and synthesizes
    each translated segment via ``POST /v1/audio/speech``. Supports zero-shot
    voice cloning through the configured reference audio.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="kakure_tts_"))

    def generate_sync(
        self,
        segments: list[dict],
        output_dir: Path | None = None,
    ) -> list[TTSResult]:
        """Generate Chinese TTS audio for all translated segments via audiocpp.

        Args:
            segments: List of dicts with 'id', 'translated_text', 'start', 'end'.
            output_dir: Directory to save audio files. Defaults to temp dir.

        Returns:
            List of TTSResult with audio paths and durations.
        """
        if output_dir is None:
            output_dir = self._temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        base_url = _SERVER.ensure(self.settings)

        # Resolve the voice preset id (matches server.json model id).
        model_id = "kakure-tts"

        logger.info("Generating TTS for %d segments via audiocpp", len(segments))
        results: list[TTSResult] = []

        for seg in segments:
            text = seg["translated_text"]
            if not text or not text.strip():
                logger.warning("Skipping empty segment %d", seg["id"])
                continue

            output_path = output_dir / f"segment_{seg['id']:04d}.wav"

            # Reuse an already-synthesized segment (checkpoint resume / crash recovery)
            if output_path.exists() and output_path.stat().st_size > 0:
                duration_ms = self._duration_ms(output_path)
                results.append(
                    TTSResult(segment_id=seg["id"], audio_path=output_path, duration_ms=duration_ms)
                )
                logger.debug("Reusing existing TTS segment %d (%dms)", seg["id"], duration_ms)
                continue

            try:
                self._synthesize(base_url, model_id, text, output_path)
                duration_ms = self._duration_ms(output_path)
                results.append(
                    TTSResult(segment_id=seg["id"], audio_path=output_path, duration_ms=duration_ms)
                )
                logger.debug("TTS segment %d: %dms, %s", seg["id"], duration_ms, text[:30])
            except Exception as e:
                logger.error("Failed to generate TTS for segment %d: %s", seg["id"], e)
                continue

        logger.info("audiocpp generation complete: %d segments", len(results))
        return results

    def _synthesize(self, base_url: str, model_id: str, text: str, output_path: Path) -> None:
        payload: dict = {
            "model": model_id,
            "input": text,
            "language": self.settings.audiocpp_language,
            "speed": self.settings.audiocpp_speed,
        }
        if self.settings.audiocpp_max_tokens and self.settings.audiocpp_max_tokens > 0:
            payload["max_tokens"] = self.settings.audiocpp_max_tokens

        r = httpx.post(
            f"{base_url}/v1/audio/speech",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"audiocpp /v1/audio/speech failed ({r.status_code}): {r.text[:500]}"
            )
        output_path.write_bytes(r.content)
        if output_path.stat().st_size == 0:
            raise RuntimeError("audiocpp returned an empty audio response.")

    @staticmethod
    def _duration_ms(path: Path) -> int:
        audio = AudioSegment.from_file(str(path))
        return len(audio)


def create_tts_processor(settings: Settings | None = None) -> BaseTTSProcessor:
    """Factory function to create the TTS processor based on settings.

    Args:
        settings: Application settings. Uses defaults if None.

    Returns:
        An AudioCppTTSProcessor instance.
    """
    settings = settings or Settings()
    if settings.tts_backend != TTSBackend.AUDIOCPP:
        logger.warning(
            "Unknown tts_backend %r; falling back to audiocpp.", settings.tts_backend
        )
    return AudioCppTTSProcessor(settings)
