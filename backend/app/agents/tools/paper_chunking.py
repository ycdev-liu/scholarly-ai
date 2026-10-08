"""按论文章节和段落生成可追溯的 Token 限长片段。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d+){0,3}|[A-D](?:\.\d+){1,3})\.?\s+"
    r"(?P<title>[^.!?。！？]{2,90})$"
)
_CHINESE_MAIN_HEADING = re.compile(
    r"^(?:第[一二三四五六七八九十]{1,3}章\s*|[一二三四五六七八九十]{1,3}[、.．]\s*)"
    r"(?P<title>[^.!?。！？]{2,60})$"
)
_CHINESE_SUB_HEADING = re.compile(
    r"^[（(][一二三四五六七八九十]{1,3}[）)]\s*(?P<title>[^.!?。！？]{2,60})$"
)
_STANDALONE_HEADING = re.compile(
    r"^(?:abstract|introduction|conclusions?|discussion|related work|methods?|materials and methods|"
    r"experiments?|results?|evaluation|limitations|future work|acknowledg(?:e)?ments?|"
    r"appendix(?:\s+[A-Z])?|摘要|引言|结论|讨论|致谢|附录(?:\s*[A-Z])?)$",
    re.I,
)
_SENTENCE_END = re.compile(r"[.!?。！？][\]\)\"'”’]*$")
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class _Paragraph:
    text: str
    page: int
    section_path: str
    metadata: dict


def _heading(line: str) -> tuple[int, str] | None:
    """识别独立标题行，并返回章节层级与标题。"""
    line = line.strip()
    if _STANDALONE_HEADING.fullmatch(line):
        return 1, line
    if _CHINESE_MAIN_HEADING.fullmatch(line):
        return 1, line
    if _CHINESE_SUB_HEADING.fullmatch(line):
        return 2, line
    match = _NUMBERED_HEADING.fullmatch(line)
    if not match:
        return None
    number = match.group("number")
    first = number.split(".", 1)[0]
    if first.isdigit() and int(first) > 20:
        return None
    title = match.group("title").strip()
    if len(title.split()) > 12 or "=" in title:
        return None
    return number.count(".") + 1, f"{number} {title}"


def _paragraphs(pages: Iterable[Document]) -> list[_Paragraph]:
    """按页扫描标题和段落；同一章节可延续到下一页。"""
    sections: dict[int, str] = {}
    result: list[_Paragraph] = []
    previous_page: int | None = None
    for page in pages:
        page_number = int(page.metadata.get("page", 0))
        if previous_page is not None and page_number > previous_page + 1:
            # 中间页面被正文清理流程过滤后，不沿用前一节的标题。
            sections = {1: "Unclassified Section"}
        previous_page = page_number
        lines: list[str] = []

        def flush() -> None:
            if lines:
                path = " > ".join(sections[level] for level in sorted(sections)) or "Front Matter"
                result.append(
                    _Paragraph(" ".join(lines).strip(), page_number, path, dict(page.metadata))
                )
                lines.clear()

        for raw_line in page.page_content.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            heading = _heading(line)
            if heading:
                flush()
                level, title = heading
                sections = {depth: name for depth, name in sections.items() if depth < level}
                sections[level] = title
                continue
            if lines and lines[-1].endswith("-"):
                lines[-1] = lines[-1][:-1] + line
            else:
                lines.append(line)
            # PDF 常丢失空行，完整句末作为保守的段落候选边界。
            if _SENTENCE_END.search(line):
                flush()
        flush()
    return result


def _make_chunk(paragraphs: list[_Paragraph], max_tokens: int) -> Document:
    """合并连续段落，并记录正文覆盖的起止页（页码从 0 开始）。"""
    path = paragraphs[0].section_path
    text = f"{path}\n\n" + "\n\n".join(part.text for part in paragraphs)
    metadata = dict(paragraphs[0].metadata)
    pages = [part.page for part in paragraphs]
    metadata.update(
        {
            "source": str(metadata.get("source", "")),
            "page": min(pages),
            "page_start": min(pages),
            "page_end": max(pages),
            "section_path": path,
            "section_title": path.split(" > ")[-1],
            "chunking_strategy": "section_paragraph_token",
            "token_count": len(_ENCODING.encode(text)),
        }
    )
    preview = next(
        (
            part.metadata.get("preview_path")
            for part in paragraphs
            if part.metadata.get("preview_path")
        ),
        None,
    )
    if preview:
        metadata["preview_path"] = preview
    else:
        metadata.pop("preview_path", None)
    if metadata["token_count"] > max_tokens:
        raise ValueError("片段超过 Token 上限，请检查标题长度或切分规则")
    return Document(page_content=text, metadata=metadata)


def split_paper_pages(
    pages: list[Document], max_tokens: int = 512, overlap_tokens: int = 64
) -> list[Document]:
    """先识别章节，再按段落打包；超长段落用 Token 窗口切分。"""
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("max_tokens 必须大于 0，overlap_tokens 必须在 0 到 max_tokens 之间")

    result: list[Document] = []
    current: list[_Paragraph] = []

    def flush(keep_overlap: bool = True) -> None:
        nonlocal current
        if not current:
            return
        result.append(_make_chunk(current, max_tokens))
        if not keep_overlap:
            current = []
            return
        tail: list[_Paragraph] = []
        total = 0
        for part in reversed(current):
            tokens = len(_ENCODING.encode(part.text))
            if total + tokens > overlap_tokens:
                break
            tail.insert(0, part)
            total += tokens
        current = tail

    for part in _paragraphs(pages):
        if current and current[-1].section_path != part.section_path:
            flush(keep_overlap=False)
        prefix = f"{part.section_path}\n\n"
        body_limit = max_tokens - len(_ENCODING.encode(prefix))
        if body_limit <= 0:
            raise ValueError("max_tokens 太小，无法容纳章节路径")
        part_tokens = _ENCODING.encode(part.text)
        if len(part_tokens) > body_limit:
            flush(keep_overlap=False)
            # 递归分隔符优先保留词句边界，长度函数和重叠量按 Token 计算。
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=body_limit,
                chunk_overlap=min(overlap_tokens, body_limit - 1),
            )
            for text in splitter.split_text(part.text):
                if text:
                    result.append(
                        _make_chunk(
                            [_Paragraph(text, part.page, part.section_path, part.metadata)],
                            max_tokens,
                        )
                    )
            continue
        proposed = current + [part]
        if (
            current
            and len(_ENCODING.encode(f"{prefix}" + "\n\n".join(p.text for p in proposed)))
            > max_tokens
        ):
            flush()
            proposed = current + [part]
            if (
                len(_ENCODING.encode(f"{prefix}" + "\n\n".join(p.text for p in proposed)))
                > max_tokens
            ):
                current = []
                proposed = [part]
        current = proposed
    flush(keep_overlap=False)
    return result
