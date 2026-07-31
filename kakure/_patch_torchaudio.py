"""Monkey-patch torchaudio.load/save to use soundfile instead of torchcodec.

torchaudio 2.7+ unconditionally requires torchcodec for all I/O, which in turn
requires FFmpeg shared libraries (DLLs on Windows). This patch avoids that
dependency chain by routing WAV I/O through soundfile (libsndfile).

Applied on import: add ``import kakure._patch_torchaudio`` to any module that
needs patched torchaudio I/O before importing torchaudio.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches():
    """Replace torchaudio.load and torchaudio.save with soundfile-backed versions."""
    global _patches_applied
    if _patches_applied:
        return

    import numpy as np
    import soundfile
    import torch
    import torchaudio

    logger.debug("Patching torchaudio I/O to use soundfile backend")

    _original_load = torchaudio.load
    _original_save = torchaudio.save
    _original_info = getattr(torchaudio, "info", None)

    def _patched_load(uri, *args, **kwargs):
        data, sample_rate = soundfile.read(uri, dtype="float32")
        if data.ndim == 1:
            data = torch.from_numpy(data.copy()).unsqueeze(0)
        else:
            data = torch.from_numpy(data.T.copy())
        return data, sample_rate

    def _patched_save(uri, src, sample_rate, **kwargs):
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        elif isinstance(src, np.ndarray):
            pass
        else:
            src = np.array(src)
        if src.ndim == 1:
            src = src.reshape(1, -1)
        soundfile.write(uri, src.T, sample_rate)

    torchaudio.load = _patched_load
    torchaudio.save = _patched_save

    _patches_applied = True
    logger.debug("Torchaudio I/O patched successfully")


# Auto-apply on import
apply_patches()
