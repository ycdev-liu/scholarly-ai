"""Auto 模式的 LangGraph 编排图。

直接调用现有工具函数，不在三个旧 Agent 之间 handoff。
通用主题走 Search_ArXiv；只有明确提到 OpenReview、审稿或投稿时走 OpenReview_Search。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph

from agents.arxiv_relaxation import collect_arxiv_candidates
from agents.query_normalization import normalize_search_query, phrase_fallback
from agents.result_ranking import dedupe_papers, rank_search_results
from agents.tools.openreview import (
    list_downloaded_papers_func,
    openreview_search_func,
    search_arxiv_func,
)
from agents.tools.vector_db import database_search_func

LocalStatus = Literal["SUFFICIENT", "INSUFFICIENT", "MISS", "FAILED"]

# 第一阶段临时规则：这些词表示用户要的是新论文，跳过本地库。
_FRESHNESS_RE = re.compile(r"最新|最近|latest|recent|2026", re.IGNORECASE)
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}")
_STOP_TERMS = {
    "帮我",
    "一下",
    "一些",
    "相关",
    "论文",
    "请帮",
    "关于",
    "哪些",
    "什么",
    "the",
    "and",
    "for",
    "papers",
    "paper",
    "some",
    "find",
    "about",
}
_DB_MISS_MARKERS = ("No relevant documents found",)
_DB_FAIL_MARKERS = ("Database search error",)

logger = logging.getLogger(__name__)


class AutoState(MessagesState, total=False):
    query: str
    skip_local: bool
    local_status: LocalStatus
    local_papers_raw: str
    local_db_text: str
    external_raw: str
    used_external: bool


_OPENREVIEW_SOURCE_RE = re.compile(
    r"openreview|conference review|\bsubmission\b|审稿|投稿",
    re.IGNORECASE,
)


def to_arxiv_query(query: str) -> str:
    """固定短语 fallback。主路径是 normalize_search_query。"""
    return phrase_fallback(query)


def select_external_source(query: str) -> Literal["arxiv", "openreview"]:
    """外部来源的轻量规则，不是 LLM Router。通用主题默认 arXiv。"""
    if _OPENREVIEW_SOURCE_RE.search(query or ""):
        return "openreview"
    return "arxiv"


def requires_fresh_search(query: str) -> bool:
    """时效性问题直接走外部搜索。这是轻量规则，不是 LLM Router。"""
    return bool(_FRESHNESS_RE.search(query or ""))


def _content_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in _TERM_RE.findall(query or ""):
        term = match.lower()
        if term in _STOP_TERMS:
            continue
        terms.append(term)
    return terms


def _papers_have_content(papers_payload: str | dict[str, Any] | None) -> tuple[bool, str]:
    """返回 (是否有本地文件, 用于重叠判断的文件名文本)。"""
    if papers_payload is None:
        return False, ""
    data: Any = papers_payload
    if isinstance(papers_payload, str):
        try:
            data = json.loads(papers_payload)
        except json.JSONDecodeError:
            return False, ""
    if not isinstance(data, dict) or not data.get("success", True):
        return False, ""
    papers = data.get("papers") or []
    names = [str(item.get("filename") or "") for item in papers if isinstance(item, dict)]
    return bool(names), "\n".join(names)


def _database_failed(database_text: str | None) -> bool:
    """Database_Search 失败时返回字符串，不抛异常。"""
    return any(marker in (database_text or "") for marker in _DB_FAIL_MARKERS)


def _database_has_content(database_text: str | None) -> bool:
    text = (database_text or "").strip()
    if not text or _database_failed(text):
        return False
    return not any(marker in text for marker in _DB_MISS_MARKERS)


def evaluate_local_results(
    query: str,
    papers_payload: str | dict[str, Any] | None,
    database_text: str | None,
) -> LocalStatus:
    """判断本地检索是否够用。

    这是第一阶段的词重叠 heuristic，后续应替换为 similarity score 或 reranker。
    Graph Node 只消费返回值，不在节点里重写判断。
    Database_Search 出错时返回 "Database search error: ..."，这是 FAILED，不是 MISS。
    """
    if _database_failed(database_text):
        return "FAILED"

    has_files, filenames = _papers_have_content(papers_payload)
    has_chunks = _database_has_content(database_text)
    if not has_files and not has_chunks:
        return "MISS"

    haystack = f"{filenames}\n{database_text or ''}".lower()
    terms = _content_terms(query)
    if terms and any(term in haystack for term in terms):
        return "SUFFICIENT"
    return "INSUFFICIENT"


def _last_user_text(state: AutoState) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _tool_pair(name: str, args: dict[str, Any], result: str) -> list[AIMessage | ToolMessage]:
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        ),
        ToolMessage(content=result, tool_call_id=call_id, name=name),
    ]


def prepare(state: AutoState) -> dict[str, Any]:
    query = _last_user_text(state)
    return {
        "query": query,
        "skip_local": requires_fresh_search(query),
        "used_external": False,
        "local_status": "MISS",
        "external_raw": "",
    }


def route_after_prepare(state: AutoState) -> Literal["local_search", "external_search"]:
    if state.get("skip_local"):
        return "external_search"
    return "local_search"


def local_search(state: AutoState) -> dict[str, Any]:
    query = state.get("query") or ""
    papers_raw = list_downloaded_papers_func()
    try:
        db_text = database_search_func(query)
    except Exception as exc:
        db_text = f"Database search error: {exc}"
    status = evaluate_local_results(query, papers_raw, db_text)
    logger.info("[LOCAL] status=%s", status)
    messages = [
        *_tool_pair("List_Downloaded_Papers", {}, papers_raw),
        *_tool_pair("Database_Search", {"query": query}, db_text),
    ]
    if status == "FAILED":
        messages.append(AIMessage(content="Local Search → FAILED → fallback to external search"))
    return {
        "messages": messages,
        "local_status": status,
        "local_papers_raw": papers_raw,
        "local_db_text": db_text,
    }


def route_after_local(state: AutoState) -> Literal["final_answer", "external_search"]:
    if state.get("local_status") == "SUFFICIENT":
        return "final_answer"
    return "external_search"


def external_search(state: AutoState) -> dict[str, Any]:
    """外部 fallback。通用主题用 Search_ArXiv，OpenReview/投稿类问题用 OpenReview_Search。"""
    query = state.get("query") or ""
    search_query = normalize_search_query(query)
    if select_external_source(query) == "openreview":
        raw = openreview_search_func(keyword=search_query, max_papers=8)
        messages = _tool_pair("OpenReview_Search", {"keyword": search_query, "max_papers": 8}, raw)
    else:
        raw = _collect_arxiv(search_query)
        messages = _tool_pair("Search_ArXiv", {"query": search_query, "max_results": 8}, raw)
    ranked_raw = _ranked_external(search_query, raw)
    return {
        "messages": messages,
        "external_raw": ranked_raw,
        "used_external": True,
    }


def _papers_from_arxiv(expression: str, max_results: int = 8) -> list[dict]:
    raw = search_arxiv_func(query=expression, max_results=max_results)
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return []
    return [paper for paper in (data.get("papers") or []) if isinstance(paper, dict)]


def _collect_arxiv(search_query: str) -> str:
    papers, attempts = collect_arxiv_candidates(search_query, _papers_from_arxiv, min_candidates=5)
    for expression, count in attempts:
        logger.info("[ARXIV] query=%s candidates=%s", expression, count)
    return json.dumps(
        {
            "total_papers": len(papers),
            "papers": papers,
            "attempts": [{"query": expression, "candidates": count} for expression, count in attempts],
        },
        ensure_ascii=False,
    )


def _ranked_external(query: str, raw: str) -> str:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    papers = [paper for paper in (data.get("papers") or []) if isinstance(paper, dict)]
    unique = dedupe_papers(papers)
    ranked = rank_search_results(query, papers, top_k=5)
    logger.info(
        "[RANK] candidates=%s deduped=%s kept=%s",
        len(papers),
        len(unique),
        len(ranked),
    )
    return json.dumps(
        {"total_papers": len(ranked), "papers": ranked},
        ensure_ascii=False,
    )


def _format_external(raw: str) -> str:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return "外部检索没有返回可解析的论文列表。"
    papers = data.get("papers") or []
    if not papers:
        return "外部检索没有找到足够相关的论文。"
    lines = ["找到以下论文："]
    for index, paper in enumerate(papers[:8], start=1):
        if not isinstance(paper, dict):
            continue
        title = paper.get("title") or "Untitled"
        url = paper.get("openreview_url") or paper.get("pdf_url") or ""
        lines.append(f"{index}. {title}" + (f"\n{url}" if url else ""))
    return "\n".join(lines)


def final_answer(state: AutoState) -> dict[str, Any]:
    if state.get("used_external"):
        content = _format_external(state.get("external_raw") or "")
    elif state.get("local_status") == "SUFFICIENT":
        snippet = (state.get("local_db_text") or "").strip()
        content = "本地已有足够相关的论文，可以直接基于本地资料回答。"
        if snippet and not any(marker in snippet for marker in _DB_MISS_MARKERS):
            content = f"{content}\n\n{snippet[:1200]}"
    else:
        content = "没有找到足够相关的论文。"
    return {"messages": [AIMessage(content=content)]}


builder = StateGraph(AutoState)
builder.add_node("prepare", prepare)
builder.add_node("local_search", local_search)
builder.add_node("external_search", external_search)
builder.add_node("final_answer", final_answer)
builder.set_entry_point("prepare")
builder.add_conditional_edges(
    "prepare",
    route_after_prepare,
    {"local_search": "local_search", "external_search": "external_search"},
)
builder.add_conditional_edges(
    "local_search",
    route_after_local,
    {"final_answer": "final_answer", "external_search": "external_search"},
)
builder.add_edge("external_search", "final_answer")
builder.add_edge("final_answer", END)

auto_supervisor = builder.compile()
