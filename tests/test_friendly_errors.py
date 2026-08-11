"""Tests for the friendly error mapping used by the API job runners."""

from __future__ import annotations

import httpx
import openai
import pytest

from kakure.api import _friendly_error

_openai_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_openai_response = httpx.Response(200, request=_openai_request)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ModuleNotFoundError("No module named 'demucs'", name="demucs"),
            "Demucs",
        ),
        (
            ModuleNotFoundError("No module named 'indextts'", name="indextts"),
            "IndexTTS",
        ),
        (
            ModuleNotFoundError("No module named 'torch'", name="torch"),
            "kotoba-whisper",
        ),
        (
            FileNotFoundError("[WinError 2] The system cannot find the file specified"),
            "ffmpeg",
        ),
        (
            openai.AuthenticationError(
                "Incorrect API key provided", response=_openai_response, body={}
            ),
            "API key",
        ),
        (
            openai.RateLimitError("Rate limit reached", response=_openai_response, body={}),
            "rate limit",
        ),
        (
            openai.APIConnectionError(request=_openai_request),
            "internet connection",
        ),
        (
            ValueError("OpenAI API key required"),
            "API key",
        ),
        (
            ValueError("Something unexpected broke"),
            "Something unexpected broke",
        ),
    ],
)
def test_friendly_error_maps(exc: BaseException, expected: str) -> None:
    message = _friendly_error(exc)
    assert expected.lower() in message.lower()
