"""Kakure - ASMR Japanese-to-Chinese bilingual voice overlay tool."""

__version__ = "0.1.0"

import kakure._patch_torchaudio  # noqa: F401, E402 — must patch before torchaudio imports
