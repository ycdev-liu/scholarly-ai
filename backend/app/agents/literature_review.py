"""Bounded literature-review workflow with evidence and approval gates."""
from __future__ import annotations
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from agents.lazy_agent import LazyLoadingAgent
from agents.query_normalization import normalize_search_query
from agents.skills import discover_skills, read_skill

from agents.tools.openreview import openreview_search_func, search_arxiv_func
from agents.tools.utils import DOWNLOAD_PAPERS_DIR
from agents.tools.vector_db import create_vector_db_from_pdf_func, database_search_func
from core import get_model, settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt

logger = logging.getLogger(__name__)
MAX_PAPERS = 8
MAX_SEARCHES = 3
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")

OPENREVIEW_ID_RE = re.compile(
    r"(?:openreview\.net/forum\?id=|openreview\s*(?:id\s*[=:]|[:：])\s*)([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
PDF_PATH_RE = re.compile(r"(?:\./)?data/downloads/papers/[^\n]+?\.pdf")


# 交流的常见状态信息
class ReviewState(MessagesState, total=False):
    query: str
    plan: list[str]
    actions: list[dict[str, str]]
    action_index: int
    action_results: list[str]
    downloaded_path: str
    evidence: list[dict[str, str]]
    local_context: str
    search_errors: list[str]


class LiteratureReviewAgent(LazyLoadingAgent):
    def __init__(self) -> None:
        super().__init__()
        # skills 列表
        self.skills = {}
        self.mcp_tools = []
        self.mcp_status: dict[str, str] = {}

    async def load(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self.skills = discover_skills()
        connections = settings.MCP_RESEARCH_SERVERS
        allowed = settings.MCP_RESEARCH_ALLOWED_TOOLS
        self.mcp_tools = []
        self.mcp_status = {}

        for name, connection in connections.items():
            if connection.get("transport") != "stdio":
                self.mcp_status[name] = "unsupported transport"
                continue
            permitted = set(allowed.get(name, []))
            if not permitted:
                self.mcp_status[name] = "no tools allowed"
                continue
            try:
                client = MultiServerMCPClient({name: connection})
                tools = await asyncio.wait_for(client.get_tools(), timeout=15)
                selected = 0
                for tool in tools:
                    if tool.name in permitted and "search" in tool.name.lower():
                        tool.name = f"{name}__{tool.name}"
                        self.mcp_tools.append(tool)
                        selected += 1
                self.mcp_status[name] = f"connected: {selected} search tools"
            except Exception as exc:
                logger.warning("MCP server %s unavailable: %s", name, exc)
                self.mcp_status[name] = f"unavailable: {type(exc).__name__}"
        self._graph = _build_graph(self)
        self._loaded = True

    def get_graph(self):
        if not self._loaded or self._graph is None:
            raise RuntimeError("Literature review agent is not loaded")
        return self._graph


def _last_query(state: ReviewState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _prepare(state: ReviewState) -> dict[str, Any]:
    query = _last_query(state)
    actions: list[dict[str, str]] = []
    notes: list[str] = []
    if re.search(r"下载|download", query, re.IGNORECASE):
        arxiv = ARXIV_ID_RE.search(query)
        openreview = OPENREVIEW_ID_RE.search(query)
        if arxiv:
            actions.append({"kind": "download_arxiv", "target": arxiv.group()})
        elif openreview:
            actions.append({"kind": "download_openreview", "target": openreview.group(1)})
        else:
            notes.append("下载需要提供 arXiv ID 或 OpenReview 论文 ID。")
    if re.search(r"建库|索引|向量化|index\s+(?:the\s+)?pdf", query, re.IGNORECASE):
        match = PDF_PATH_RE.search(query)
        if match:
            actions.append({"kind": "index", "target": match.group()})
        elif actions and actions[-1]["kind"].startswith("download_"):
            actions.append({"kind": "index", "target": "__downloaded__"})
        else:
            notes.append("建库需要提供已下载论文的 PDF 路径。")
    return {
        "query": query,
        "actions": actions,
        "action_index": 0,
        "action_results": notes,
        "downloaded_path": "",
        "plan": ["检索论文", "核对来源与证据", "比较方法和局限", "撰写综述"],
        "evidence": [],
        "search_errors": [],
    }


def _after_prepare(state: ReviewState) -> str:
    return "approval" if state.get("actions") else "search"


def _approve(state: ReviewState) -> dict[str, Any]:
    action = dict(state["actions"][state["action_index"]])
    if action["target"] == "__downloaded__":
        action["target"] = state.get("downloaded_path") or "刚下载的论文 PDF"
    decision = interrupt({"kind": "approval_required", "action": action})
    if decision not in ("approve", "deny"):
        decision = "deny"
    return {"action_results": [*state.get("action_results", []), decision]}


def _after_approval(state: ReviewState) -> str:
    return "execute" if state["action_results"][-1] == "approve" else "advance"


async def _execute(state: ReviewState) -> dict[str, Any]:
    action = state["actions"][state["action_index"]]

    downloaded_path = state.get("downloaded_path", "")
    if action["kind"].startswith("download_"):
        from agents.tools.openreview import download_paper_from_arxiv_func, download_paper_func

        if action["kind"] == "download_openreview":
            result = await asyncio.to_thread(download_paper_func, paper_id=action["target"])
        else:
            result = await asyncio.to_thread(download_paper_from_arxiv_func, arxiv_id=action["target"])
        try:
            payload = json.loads(result)
            if payload.get("success"):
                downloaded_path = str(payload.get("file_path") or "")
        except (TypeError, ValueError, AttributeError):
            pass
    else:
        target = downloaded_path if action["target"] == "__downloaded__" else action["target"]
        path = Path(target).resolve()
        root = Path(DOWNLOAD_PAPERS_DIR).resolve()

        if not path.is_relative_to(root) or not path.is_file():
            result = "PDF must exist in the downloaded papers directory."
        else:
            result = await asyncio.to_thread(create_vector_db_from_pdf_func, str(path))
    return {"action_results": [*state["action_results"], str(result)], "downloaded_path": downloaded_path}


def _advance(state: ReviewState) -> dict[str, int]:
    next_index = state["action_index"] + 1
    if next_index < len(state["actions"]):
        next_action = state["actions"][next_index]
        if next_action["target"] == "__downloaded__" and not state.get("downloaded_path"):
            next_index += 1
    return {"action_index": next_index}


def _after_advance(state: ReviewState) -> str:
    return "approval" if state["action_index"] < len(state["actions"]) else "search"


def _extract_papers(raw: str, source: str) -> list[dict[str, str]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    found = []
    for paper in data.get("papers", [])[:MAX_PAPERS]:
        if not isinstance(paper, dict):
            continue
        url = paper.get("openreview_url") or paper.get("pdf_url") or paper.get("url")
        title = paper.get("title")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")) or not title:
            continue
        found.append({
            "title": str(title), "url": url,
            "abstract": str(paper.get("abstract") or "")[:900], "source": source,
        })
    return found


async def _search(state: ReviewState, owner: LiteratureReviewAgent) -> dict[str, Any]:
    query = state["query"]
    search_query = await asyncio.to_thread(normalize_search_query, query)
    errors = []
    try:
        local = await asyncio.to_thread(database_search_func, query)
    except Exception as exc:
        local = ""
        errors.append(f"local search: {type(exc).__name__}")
    if str(local).startswith("Database search error"):
        errors.append("local search: unavailable")
        local = ""
    searches: list[tuple[str, Any]] = [("arxiv", lambda: search_arxiv_func(search_query, MAX_PAPERS))]
    if "openreview" in query.lower() or "会议" in query:
        searches.append(("openreview", lambda: openreview_search_func(keyword=search_query, max_papers=MAX_PAPERS)))
    papers = []
    for source, func in searches[:MAX_SEARCHES]:
        try:
            papers.extend(_extract_papers(await asyncio.to_thread(func), source))
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}")
    for tool in owner.mcp_tools[: max(0, MAX_SEARCHES - len(searches))]:
        try:
            args = tool.args
            key = "query" if "query" in args else "keyword" if "keyword" in args else None
            if key:
                result = await asyncio.wait_for(tool.ainvoke({key: search_query}), timeout=20)
                papers.extend(_extract_papers(str(result), tool.name))
        except Exception as exc:
            errors.append(f"{tool.name}: {type(exc).__name__}")
    unique: dict[str, dict[str, str]] = {}
    for paper in papers:
        unique.setdefault(paper["url"], paper)
    return {
        "evidence": list(unique.values())[:MAX_PAPERS],
        "local_context": str(local)[:2000],
        "search_errors": errors,
    }


async def _write(state: ReviewState, owner: LiteratureReviewAgent, config: Any) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    if not evidence:
        content = "没有找到带可验证来源链接的论文，暂时无法生成可靠的文献综述。"
    else:
        skill_names = ["literature-review"]
        if len(evidence) >= 2:
            skill_names.append("evidence-check")
        instructions = "\n\n".join(read_skill(name, owner.skills) for name in skill_names)
        records = "\n".join(
            f"[{i}] {paper['title']} | {paper['abstract']}"
            for i, paper in enumerate(evidence, 1)
        )
        model_name = (config.get("configurable") or {}).get("model", settings.DEFAULT_MODEL)
        model = get_model(model_name)
        response = await model.ainvoke([
            SystemMessage(content=(
                "Write an academic literature review in the user's language. Compare approaches, "
                "findings and limitations. Cite only numbered records [1], [2], etc. "
                "Do not output any URL or claim facts not in the records. "
                "Local excerpts may suggest terminology, but are not citable evidence.\n" + instructions
            )),
            HumanMessage(content=(
                f"Question: {state['query']}\nEvidence:\n{records}"
                f"\nLocal excerpts (uncited): {state.get('local_context', '')[:1200]}"
            )),
        ], config=config)
        body = str(response.content)
        body = re.sub(r"https?://\S+", "", body)
        body = re.sub(
            r"\[(\d+)\]",
            lambda match: match.group() if 1 <= int(match.group(1)) <= len(evidence) else "",
            body,
        )
        sources = "\n".join(f"[{i}] [{p['title']}]({p['url']})" for i, p in enumerate(evidence, 1))
        qualifier = "\n\n现有证据不足以充分比较不同方法。" if len(evidence) < 2 else ""
        content = f"{body}{qualifier}\n\n来源：\n{sources}"
    if state.get("search_errors"):
        content += "\n\n部分检索源不可用：" + ", ".join(state["search_errors"])
    for action_result in state.get("action_results", []):
        if action_result == "deny":
            content += "\n\n已按要求取消敏感操作。"
        elif action_result != "approve":
            content += f"\n\n操作结果：{action_result[:500]}"
    return {"messages": [AIMessage(content=content)]}


def _build_graph(owner: LiteratureReviewAgent):
    async def search_node(state: ReviewState):
        return await _search(state, owner)

    async def write_node(state: ReviewState, config: Any):
        return await _write(state, owner, config)

    graph = StateGraph(ReviewState)
    graph.add_node("prepare", _prepare)
    graph.add_node("approval", _approve)
    graph.add_node("execute", _execute)
    graph.add_node("advance", _advance)
    graph.add_node("search", search_node)
    graph.add_node("write", write_node)
    graph.set_entry_point("prepare")
    graph.add_conditional_edges("prepare", _after_prepare, {"approval": "approval", "search": "search"})
    graph.add_conditional_edges("approval", _after_approval, {"execute": "execute", "advance": "advance"})
    graph.add_edge("execute", "advance")
    graph.add_conditional_edges("advance", _after_advance, {"approval": "approval", "search": "search"})
    graph.add_edge("search", "write")
    graph.add_edge("write", END)
    return graph.compile()


literature_review_agent = LiteratureReviewAgent()
