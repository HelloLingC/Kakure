"""Translation module - Japanese to Chinese translation."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kakure.config import Settings, TranslationBackend

logger = logging.getLogger(__name__)

# Rough chars -> tokens estimate for Japanese/Chinese text.
# Japanese text averages ~1.2 tokens per char (kana/kanji).
_TOKEN_ESTIMATE_FACTOR = 1.2

# Regex to extract the first JSON object from a response that might contain
# stray prose around the JSON.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_batch_json(raw: str, items: list[tuple[int, str]]) -> dict[int, str]:
    """Parse an LLM batch response into a {id: translation} dict.

    Accepts both `{"translations":[{"id":..,"zh":..}]}` JSON objects and
    attempts to recover from malformed wrapping. Raises ValueError on failure.
    """
    text = raw.strip()
    # Try direct json.loads first
    data: object
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise ValueError(f"Batch response has no parseable JSON: {raw!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Batch response JSON parse failed: {e}; raw={raw!r}") from e

    translations: list
    if isinstance(data, dict) and "translations" in data:
        translations = data["translations"]
    elif isinstance(data, list):
        translations = data
    else:
        raise ValueError(f"Batch response has unexpected shape: {raw!r}")

    out: dict[int, str] = {}
    for entry in translations:
        if not isinstance(entry, dict):
            raise ValueError(f"Batch entry is not an object: {entry!r}")
        if "id" not in entry or "zh" not in entry:
            raise ValueError(f"Batch entry missing id/zh: {entry!r}")
        out[entry["id"]] = str(entry["zh"]).strip()

    # Validate coverage
    expected_ids = {i for i, _ in items}
    missing = expected_ids - set(out.keys())
    if missing:
        # Positional fallback: some models renumber ids as 0..n-1 regardless of
        # the ids shown in the prompt. If the response ids are exactly 0..n-1
        # and the count matches, map them back to the expected ids in input order.
        if len(out) == len(items) and sorted(out.keys()) == list(range(len(items))):
            return {idx: out[pos] for pos, (idx, _) in enumerate(items)}
        raise ValueError(f"Batch response missing ids: {sorted(missing)}; raw={raw!r}")
    return out


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for Japanese/Chinese text."""
    return int(len(text) * _TOKEN_ESTIMATE_FACTOR)


def _iter_batches(
    segments: list,
    batch_size: int,
    token_limit: int,
) -> list[list]:
    """Split segments into batches bounded by max segment count and token budget.

    Yields successive sub-lists. A single segment that by itself exceeds the
    token budget still forms its own batch (length 1) rather than dropping the
    segment.
    """
    batches: list[list] = []
    current: list = []
    current_tokens = 0
    for seg in segments:
        seg_tokens = _estimate_tokens(seg.original_text)
        would_exceed_count = len(current) + 1 > batch_size
        would_exceed_tokens = current and (current_tokens + seg_tokens > token_limit)
        if current and (would_exceed_count or would_exceed_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += seg_tokens
    if current:
        batches.append(current)
    return batches


# System prompt for ASMR-aware translation
ASMR_TRANSLATION_PROMPT = """你是一位资深的 ASMR 音声与语音剧专业译员，极其擅长将日文 ASMR 台本翻译为极具临场感、亲昵感和画面感的中文。

【翻译核心要求】
1. 口语化与听觉感：译文必须极其适合口头朗读。摆脱任何书面化或“翻译腔”，使用自然、地道、带有些许呼吸感和温度感的中文表达。
2. 拟声词与情感词处理：
   - 将日文拟声词/拟态词（如 くんくん、ちゅっ、ふわふわ 等）转换为富有表现力的中文听觉描述或拟声词（如“嗅嗅”、“啾”、“软绵绵”）。
3. 人设与语调契合：
   - 严格根据上下文判断角色的性格（如：傲娇、温柔治愈、病娇、小恶魔、主仆等）。
   - 调整中文的称呼与句尾语气词（如：呢、呀、哦、嘛、～），精准还原日文敬语与亲疏关系。
4. 意译优先于直译：遇到日式习惯表达时，优先转换为中文台词中同等情感浓度的表达，确保听众能产生强烈的代入感。

请直接输出翻译后的中文台本，保持原文本的段落结构，绝对不允许输出任何额外的解释。"""


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
        self.prompt = settings.translation_prompt or ASMR_TRANSLATION_PROMPT

    def translate(self, text: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": self.prompt},
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
        )
        choice = response.choices[0]
        result = choice.message.content
        logger.info(
            "LLM response | model=%s finish_reason=%s | text=%r",
            self.model,
            getattr(choice, "finish_reason", None),
            result,
        )
        if not result or not result.strip():
            raise ValueError(
                f"OpenAI returned empty response (finish_reason={getattr(choice, 'finish_reason', None)})"
            )
        return result.strip()

    def translate_batch(self, items: list[tuple[int, str]], context: str = "") -> dict[int, str]:
        """Translate multiple Japanese segments in one API call.

        Args:
            items: List of (id, japanese_text) pairs.
            context: Previous segments for context.

        Returns:
            Dict mapping id -> Chinese translation.

        Raises:
            ValueError: If response cannot be parsed or ids don't match.
        """
        if not items:
            return {}

        inputs_block = "\n".join(f"[{i}] {text}" for i, (idx, text) in enumerate(items))
        batch_instruction = (
            (
                "请将以下日文片段按顺序逐一翻译为中文。返回一个 JSON 对象，格式为：\n"
                '{"translations": [{"id": <序号>, "zh": "<中文译文>"}]}\n'
                "要求：\n"
                "- 共翻译 <n> 条，序号 0 到 <n-1>，与输入顺序一一对应。\n"
                "- zh 字段仅含译文本身，不含序号、解释或任何前后缀。\n"
                "- 不要遗漏或合并任何条目。\n"
                "片段：\n" + inputs_block
            )
            .replace("<n>", str(len(items)))
            .replace("<n-1>", str(len(items) - 1))
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompt},
        ]
        if context:
            messages.append({"role": "user", "content": context})
            messages.append(
                {"role": "assistant", "content": "(providing context for next translation)"}
            )
        messages.append({"role": "user", "content": batch_instruction})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        result = choice.message.content
        logger.info(
            "LLM batch response | model=%s finish_reason=%s batch=%d | text=%r",
            self.model,
            getattr(choice, "finish_reason", None),
            len(items),
            result,
        )
        if not result or not result.strip():
            raise ValueError(
                f"OpenAI returned empty batch response "
                f"(finish_reason={getattr(choice, 'finish_reason', None)})"
            )

        return _parse_batch_json(result, items)


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
                        "OpenAI API key is missing. Add it in the Settings tab "
                        "or set openai_api_key in kakure.toml."
                    )
                self._backend = OpenAITranslator(self.settings)
                logger.info(
                    "Using OpenAI translation backend (model=%s)", self.settings.openai_model
                )
            else:
                raise ValueError(
                    f"Unknown translation backend: {self.settings.translation_backend}"
                )
        return self._backend

    def translate_segment(self, segment: TranslatedSegment) -> str:
        """Translate a single segment's original text to Chinese."""
        return self.backend.translate(segment.original_text)

    def translate_segments(self, segments: list[TranslatedSegment]) -> list[TranslatedSegment]:
        """Translate all segments.

        Segments are split into batches (one API call per batch, structured
        JSON output) and dispatched concurrently through a thread pool bounded
        by ``settings.translation_max_concurrency``. Batches are independent
        (no rolling context), so they can run in parallel. On batch failure the
        failing batch falls back to per-segment single requests.

        Args:
            segments: List of segments with original text.

        Returns:
            List of segments with translated text filled in.
        """
        logger.info("Translating %d segments", len(segments))
        batch_size = getattr(self.settings, "translation_batch_size", 1) or 1
        token_limit = getattr(self.settings, "translation_batch_token_limit", 8000) or 8000
        concurrency = getattr(self.settings, "translation_max_concurrency", 4) or 1

        batches = _iter_batches(segments, batch_size, token_limit)
        if batch_size <= 1 or not hasattr(self.backend, "translate_batch"):
            self._translate_in_parallel([[seg] for seg in segments], concurrency)
            logger.info("Translation complete")
            return segments

        self._translate_in_parallel(batches, concurrency)
        logger.info("Translation complete")
        return segments

    def _translate_in_parallel(
        self,
        batches: list[list[TranslatedSegment]],
        concurrency: int,
    ) -> None:
        """Translate all batches concurrently via a bounded thread pool.

        Each batch is translated independently (no rolling context). Batches
        that fail are retried per-segment within the same pool.

        Args:
            batches: List of segment batches (each dispatched as one request).
            concurrency: Max number of concurrent API calls.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            "Translating %d batches with concurrency=%d",
            len(batches),
            max(1, concurrency),
        )
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_to_batch = {
                executor.submit(self._translate_batch_worker, batch): batch for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    results = future.result()
                except Exception as e:
                    logger.warning(
                        "Batch translate failed (%s); falling back to per-segment for %d segments",
                        e,
                        len(batch),
                    )
                    self._translate_batch_one_by_one(batch)
                    continue

                for seg in batch:
                    if seg.id in results and results[seg.id]:
                        seg.translated_text = results[seg.id]
                    else:
                        logger.warning(
                            "Batch missing translation for segment %d; using original", seg.id
                        )
                        seg.translated_text = seg.original_text
                logger.info(
                    "Translated batch of %d segments (ids=%s)",
                    len(batch),
                    [s.id for s in batch],
                )

    def _translate_batch_worker(
        self,
        batch: list[TranslatedSegment],
    ) -> dict[int, str]:
        """Translate one batch in a worker thread. Returns {id: translation}."""
        if len(batch) == 1:
            text = self.backend.translate(batch[0].original_text)
            return {batch[0].id: text}
        items = [(seg.id, seg.original_text) for seg in batch]
        return self.backend.translate_batch(items)  # type: ignore[attr-defined]

    def _translate_batch_one_by_one(self, batch: list[TranslatedSegment]) -> None:
        """Translate each segment of a batch individually (fallback path)."""
        for seg in batch:
            try:
                seg.translated_text = self.backend.translate(seg.original_text)
                logger.debug("Segment %d: %s -> %s", seg.id, seg.original_text, seg.translated_text)
            except Exception as e:
                logger.error("Failed to translate segment %d: %s", seg.id, e)
                seg.translated_text = seg.original_text  # Fallback to original
