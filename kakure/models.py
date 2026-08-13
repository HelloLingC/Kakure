"""Model management - list, download, and delete AI models (Whisper, IndexTTS).

Models are stored in the HuggingFace Hub cache (``HF_HUB_CACHE`` / ``HF_HOME``
or ``~/.cache/huggingface/hub``). This module provides the catalog of models
Kakure uses, reports their install status and disk usage, downloads them with
progress callbacks, and removes them from the cache.

Heavy third-party imports happen inside functions so importing this module
stays cheap.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Group ids used by the web UI
GROUP_WHISPER = "whisper"
GROUP_KOTOBA = "kotoba"
GROUP_INDEXTTS = "indextts"

MODEL_GROUPS: list[dict] = [
    {
        "id": GROUP_WHISPER,
        "name": "Whisper (faster-whisper) - ASR",
        "models": [
            {
                "id": "whisper-tiny",
                "name": "Whisper tiny",
                "repo_id": "Systran/faster-whisper-tiny",
                "approx_size": "~75 MB",
                "probe_file": "model.bin",
            },
            {
                "id": "whisper-base",
                "name": "Whisper base",
                "repo_id": "Systran/faster-whisper-base",
                "approx_size": "~145 MB",
                "probe_file": "model.bin",
            },
            {
                "id": "whisper-small",
                "name": "Whisper small",
                "repo_id": "Systran/faster-whisper-small",
                "approx_size": "~465 MB",
                "probe_file": "model.bin",
            },
            {
                "id": "whisper-medium",
                "name": "Whisper medium",
                "repo_id": "Systran/faster-whisper-medium",
                "approx_size": "~1.5 GB",
                "probe_file": "model.bin",
            },
            {
                "id": "whisper-large-v3",
                "name": "Whisper large-v3",
                "repo_id": "Systran/faster-whisper-large-v3",
                "approx_size": "~3.1 GB",
                "probe_file": "model.bin",
            },
            {
                "id": "whisper-distil-large-v3",
                "name": "Whisper distil-large-v3",
                "repo_id": "Systran/faster-whisper-distil-large-v3",
                "approx_size": "~1.6 GB",
                "probe_file": "model.bin",
            },
        ],
    },
    {
        "id": GROUP_KOTOBA,
        "name": "Kotoba-Whisper (Japanese ASR)",
        "models": [
            {
                "id": "kotoba-v2.0",
                "name": "Kotoba-Whisper v2.0",
                "repo_id": "kotoba-tech/kotoba-whisper-v2.0",
                "approx_size": "~1.5 GB",
                "probe_file": "model.safetensors",
            },
            {
                "id": "kotoba-v2.1",
                "name": "Kotoba-Whisper v2.1",
                "repo_id": "kotoba-tech/kotoba-whisper-v2.1",
                "approx_size": "~1.5 GB",
                "probe_file": "model.safetensors",
            },
            {
                "id": "kotoba-v2.2",
                "name": "Kotoba-Whisper v2.2",
                "repo_id": "kotoba-tech/kotoba-whisper-v2.2",
                "approx_size": "~1.5 GB",
                "probe_file": "model.safetensors",
            },
        ],
    },
    {
        "id": GROUP_INDEXTTS,
        "name": "IndexTTS (TTS)",
        "models": [
            {
                "id": "indextts-main",
                "name": "IndexTTS-2 main model",
                "repo_id": "IndexTeam/IndexTTS-2",
                "approx_size": "~6 GB",
                "probe_file": "config.yaml",
            },
            {
                "id": "indextts-w2v",
                "name": "w2v-bert-2.0 (speech encoder)",
                "repo_id": "facebook/w2v-bert-2.0",
                "approx_size": "~2.3 GB",
                "probe_file": "model.safetensors",
            },
            {
                "id": "indextts-maskgct",
                "name": "MaskGCT (semantic codec)",
                "repo_id": "amphion/MaskGCT",
                "approx_size": "~177 MB",
                "probe_file": "semantic_codec/model.safetensors",
            },
            {
                "id": "indextts-campplus",
                "name": "Cam++ (speaker embedding)",
                "repo_id": "funasr/campplus",
                "approx_size": "~28 MB",
                "probe_file": "campplus_cn_common.bin",
            },
            {
                "id": "indextts-bigvgan",
                "name": "BigVGAN (vocoder)",
                "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
                "approx_size": "~450 MB",
                "probe_file": "config.json",
            },
        ],
    },
]


def _model_by_repo(repo_id: str) -> dict | None:
    """Return the catalog entry for a repo_id, or None."""
    for group in MODEL_GROUPS:
        for model in group["models"]:
            if model["repo_id"] == repo_id:
                return model
    return None


def hf_cache_dir() -> Path:
    """Resolve the HuggingFace Hub cache directory used for downloads."""
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_sizes() -> dict[str, int]:
    """Return ``{repo_id: size_on_disk_bytes}`` for every cached model repo."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        return {repo.repo_id: repo.size_on_disk for repo in info.repos}
    except Exception as e:
        logger.warning("Failed to scan HuggingFace cache: %s", e)
        return {}


def _is_installed(repo_id: str, probe_file: str) -> bool:
    """True when ``probe_file`` is present in the local cache for ``repo_id``.

    Uses the offline cache lookup so partially-downloaded repos are reported
    as not installed.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id, probe_file, cache_dir=str(hf_cache_dir()))
        return path is not None
    except Exception:
        return False


def model_status() -> dict:
    """Build the model catalog annotated with install status and sizes.

    Returns:
        A dict with ``groups`` (catalog plus ``installed`` / ``size_on_disk``
        per model) and ``cache_dir`` (the resolved HF cache directory).
    """
    sizes = _cached_sizes()
    groups = []
    for group in MODEL_GROUPS:
        models = []
        for model in group["models"]:
            entry = dict(model)
            entry["installed"] = _is_installed(model["repo_id"], model["probe_file"])
            entry["size_on_disk"] = sizes.get(model["repo_id"])
            models.append(entry)
        groups.append({**group, "models": models})
    return {"groups": groups, "cache_dir": str(hf_cache_dir())}


def download_model(repo_id: str, on_progress: Callable[[int, int], None] | None = None) -> str:
    """Download a model into the HuggingFace cache, reporting progress.

    Args:
        repo_id: Repository to download (must be in the catalog).
        on_progress: Optional callback ``(bytes_done, bytes_total)`` invoked
            as chunks land. Bytes are cumulative across all files.

    Returns:
        The local snapshot directory.
    """
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as auto_tqdm

    if _model_by_repo(repo_id) is None:
        raise ValueError(f"Unknown model repository: {repo_id}")

    class _ProgressTqdm(auto_tqdm):
        """tqdm subclass that mirrors byte progress to ``on_progress``.

        ``snapshot_download`` creates two bars: a per-file "Fetching N files"
        bar and a shared bytes bar (``unit == "B"``) whose total grows as each
        file's size becomes known. Only the bytes bar is reported, and the
        percent is clamped because totals are discovered progressively.
        """

        def update(self, n: float | None = 1) -> None:
            super().update(n)
            if on_progress is not None and self.unit == "B" and self.total:
                on_progress(int(min(self.n, self.total)), int(self.total))

    logger.info("Downloading model '%s' into %s", repo_id, hf_cache_dir())
    return snapshot_download(repo_id, cache_dir=str(hf_cache_dir()), tqdm_class=_ProgressTqdm)


def delete_model(repo_id: str) -> int:
    """Remove a cached model from the HuggingFace cache.

    Args:
        repo_id: Repository to remove.

    Returns:
        The number of bytes freed (0 if nothing was cached).
    """
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    repo = next((r for r in info.repos if r.repo_id == repo_id), None)
    if repo is None:
        return 0

    strategy = info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
    freed = strategy.expected_freed_size
    strategy.execute()
    logger.info("Deleted cached model '%s' (freed %d bytes)", repo_id, freed)
    return freed
