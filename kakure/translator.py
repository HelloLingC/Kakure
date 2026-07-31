"""Translation module - Japanese to Chinese translation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kakure.config import Settings, TranslationBackend

logger = logging.getLogger(__name__)

# System prompt for ASMR-aware translation
ASMR_TRANSLATION_PROMPT = """You are a professional Japanese-to-Chinese translator specializing in ASMR content.

Rules:
1. Translate naturally, preserving the emotional tone and intimacy of ASMR content.
2. Keep onomatopoeia and sound effects as-is (e.g., しょぼしょぼ, ぽかぽか).
3. Use soft, gentle Chinese expressions that match the ASMR atmosphere.
4. Preserve honorifics and politeness levels in the translation tone.
5. Keep the translation concise — it will be spoken aloud, so avoid overly long sentences.
6. If a segment is just breathing, sighs, or non-verbal sounds, translate as-is or use Chinese equivalents.
7. Do NOT add explanations, notes, or brackets. Just output the translation.

Translate the following Japanese text to Chinese (Simplified):"""


@dataclass
class TranslatedSegment:
    """A segment with both original and translated text."""

    id: int
    start: float
    end: float
    original_text: str
    translated_text: str


class TranslationBackend_(ABC):
    """Abstract base class for translation backends."""

    @abstractmethod
    def translate(self, text: str, context: str = "") -> str:
        """Translate Japanese text to Chinese.

        Args:
            text: Japanese text to translate.
            context: Previous segments for context.

        Returns:
            Chinese translation.
        """
        ...


class OpenAITranslator(TranslationBackend_):
    """Translation using OpenAI GPT models."""

    def __init__(self, settings: Settings):
        import openai

        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model

    def translate(self, text: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": ASMR_TRANSLATION_PROMPT},
        ]
        if context:
            messages.append({"role": "user", "content": context})
            messages.append(
                {"role": "assistant", "content": "(providing context for next translation)"}
            )
        messages.append({"role": "user", "content": text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("OpenAI returned empty response")
        return result.strip()


class DeepLTranslator(TranslationBackend_):
    """Translation using DeepL API."""

    def __init__(self, settings: Settings):
        import deepl

        self.translator = deepl.Translator(settings.deepl_api_key)

    def translate(self, text: str, context: str = "") -> str:
        result = self.translator.translate_text(
            text,
            source_lang="JA",
            target_lang="ZH",
        )
        return result.text.strip()


class Translator:
    """Main translator that delegates to the configured backend."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._backend: TranslationBackend_ | None = None

    @property
    def backend(self) -> TranslationBackend_:
        """Lazy-initialize the translation backend."""
        if self._backend is None:
            if self.settings.translation_backend == TranslationBackend.OPENAI:
                if not self.settings.openai_api_key:
                    raise ValueError(
                        "OpenAI API key required. Set OPENAI_API_KEY environment variable."
                    )
                self._backend = OpenAITranslator(self.settings)
                logger.info("Using OpenAI translation backend (model=%s)", self.settings.openai_model)
            elif self.settings.translation_backend == TranslationBackend.DEEPL:
                if not self.settings.deepl_api_key:
                    raise ValueError(
                        "DeepL API key required. Set DEEPL_API_KEY environment variable."
                    )
                self._backend = DeepLTranslator(self.settings)
                logger.info("Using DeepL translation backend")
            else:
                raise ValueError(
                    f"Unknown translation backend: {self.settings.translation_backend}"
                )
        return self._backend

    def translate_segment(self, segment: "TranslatedSegment") -> str:
        """Translate a single segment's original text to Chinese."""
        return self.backend.translate(segment.original_text)

    def translate_segments(
        self, segments: list["TranslatedSegment"]
    ) -> list["TranslatedSegment"]:
        """Translate all segments, using context from previous segments.

        Args:
            segments: List of segments with original text.

        Returns:
            List of segments with translated text filled in.
        """
        logger.info("Translating %d segments", len(segments))
        context_parts: list[str] = []

        for i, seg in enumerate(segments):
            context = "\n".join(context_parts[-3:]) if context_parts else ""
            try:
                translated = self.backend.translate(seg.original_text, context=context)
                seg.translated_text = translated
                context_parts.append(f"JP: {seg.original_text}\nCN: {translated}")
                logger.debug("Segment %d: %s -> %s", seg.id, seg.original_text, translated)
            except Exception as e:
                logger.error("Failed to translate segment %d: %s", seg.id, e)
                seg.translated_text = seg.original_text  # Fallback to original

        logger.info("Translation complete")
        return segments