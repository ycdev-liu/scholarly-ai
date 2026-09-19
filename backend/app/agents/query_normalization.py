"""把自然语言问题收成学术检索词。

复用 core.get_model，不新建模型服务，也不新建 Agent。
模型失败或输出不合格时，退回已有固定短语，再不行就用原句。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# 只保留已有短语，作为模型失败时的 fallback，不再扩展。
_ZH_ARXIV_PHRASES = (
    ("医学图像分割", "medical image segmentation"),
    ("图像分割", "image segmentation"),
    ("医学图像", "medical image"),
    ("语义分割", "semantic segmentation"),
    ("目标检测", "object detection"),
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CHATTY_RE = re.compile(
    r"^(please|help|find|show|look|search|can you|could you|i want|i need)\b",
    re.IGNORECASE,
)
_PROMPT = (
    "Rewrite the user request into one short English academic search query "
    "for arXiv or OpenReview. Output only the keywords on a single line. "
    "No explanation, no markdown, no quotes, and no full sentence. "
    "Prefer 3 to 8 words. Do not answer the question. "
    "If the input is already a good English search query, return it unchanged."
)


def phrase_fallback(query: str) -> str:
    """固定短语替换。未命中时返回去掉空白后的原句。"""
    text = query or ""
    matched: list[str] = []
    used: list[str] = []
    for chinese, english in _ZH_ARXIV_PHRASES:
        if chinese in text and not any(chinese in longer for longer in used):
            matched.append(english)
            used.append(chinese)
    if matched:
        return " ".join(dict.fromkeys(matched))
    return text.strip()


def _already_keywords(text: str) -> bool:
    """英文本身已经是短检索词时，不再改写。"""
    if _CJK_RE.search(text) or "?" in text or "？" in text:
        return False
    if _CHATTY_RE.search(text):
        return False
    words = text.split()
    return 0 < len(words) <= 10


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _clean_keywords(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    line = line.strip("`\"'“”").lstrip("-* ").strip()
    line = re.sub(r"^(search query|query|keywords)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip(" .")
    if not line or _CJK_RE.search(line):
        return ""
    if len(line.split()) > 12:
        return ""
    if re.search(r"[^A-Za-z0-9+\-./\s]", line):
        return ""
    return line


def _invoke_model(query: str) -> str:
    from dotenv import load_dotenv

    from core import get_model, settings

    # ChatAnthropic 读的是进程环境变量。服务启动时会 load_dotenv，这里再调一次，避免只加载了 Settings。
    load_dotenv()
    model = get_model(settings.DEFAULT_MODEL)
    response = model.invoke([SystemMessage(content=_PROMPT), HumanMessage(content=query)])
    return _message_text(response)


def normalize_search_query(query: str) -> str:
    """生成适合 arXiv / OpenReview 的短英文检索词。失败时不抛出。"""
    text = (query or "").strip()
    if not text or _already_keywords(text):
        return text
    try:
        keywords = _clean_keywords(_invoke_model(text))
    except Exception:
        logger.warning("[QUERY] normalization failed, using fallback")
        keywords = ""
    if keywords:
        logger.info("[QUERY] %s -> %s", text, keywords)
        return keywords
    fallback = phrase_fallback(text)
    logger.info("[QUERY] fallback %s -> %s", text, fallback)
    return fallback
