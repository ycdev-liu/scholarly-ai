import json
import re
import sys
from pathlib import Path

import pytest
from agents import literature_review as review
from agents.skills import discover_skills, read_skill
from langchain_community.chat_models import FakeListChatModel
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


def _papers():
    return json.dumps({"papers": [
        {"title": "Study A", "abstract": "Method A improves retrieval.", "pdf_url": "https://arxiv.org/pdf/1234.5678"},
        {"title": "Study B", "abstract": "Method B has lower cost.", "pdf_url": "https://arxiv.org/pdf/1234.5679"},
    ]})


@pytest.mark.asyncio
async def test_review_uses_verified_sources(monkeypatch):
    owner = review.LiteratureReviewAgent()
    owner.skills = discover_skills()
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: _papers())
    monkeypatch.setattr(review, "database_search_func", lambda query: "local context")
    monkeypatch.setattr(review, "get_model", lambda name: FakeListChatModel(responses=["A [1] and B [2]. https://invalid.test/"]))
    graph = review._build_graph(owner)
    result = await graph.ainvoke({"messages": [HumanMessage(content="Compare retrieval methods")]})
    answer = result["messages"][-1].content
    assert "A [1] and B [2]" in answer
    assert "https://invalid.test" not in answer
    assert "https://arxiv.org/pdf/1234.5678" in answer
    assert "https://arxiv.org/pdf/1234.5679" in answer


@pytest.mark.asyncio
async def test_approval_denied_and_approved_on_same_thread(monkeypatch):
    owner = review.LiteratureReviewAgent()
    owner.skills = discover_skills()
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: _papers())
    monkeypatch.setattr(review, "database_search_func", lambda query: "")
    monkeypatch.setattr(review, "get_model", lambda name: FakeListChatModel(responses=["Review."]))
    calls = []
    from agents.tools import openreview
    monkeypatch.setattr(openreview, "download_paper_from_arxiv_func", lambda arxiv_id: calls.append(arxiv_id) or "downloaded")
    graph = review._build_graph(owner)
    graph.checkpointer = MemorySaver()
    for decision, expected in [("deny", 0), ("approve", 1)]:
        config = {"configurable": {"thread_id": decision}}
        await graph.ainvoke({"messages": [HumanMessage(content="下载 arXiv:1234.5678 并综述")]}, config)
        state = await graph.aget_state(config)
        assert state.tasks[0].interrupts[0].value["kind"] == "approval_required"
        result = await graph.ainvoke(Command(resume=decision), config)
        assert len(calls) == expected
        assert result["messages"][-1].content


def test_skills_are_discoverable_and_readable():
    registry = discover_skills()
    assert {"literature-review", "evidence-check"} <= registry.keys()
    assert "Compare" in read_skill("literature-review", registry)


@pytest.mark.asyncio
async def test_mcp_allowlist_and_unavailable_fallback(monkeypatch):
    from core import settings

    class Tool:
        name = "search_remote"

    class Client:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self):
            if "broken" in self.connections:
                raise OSError("offline")
            return [Tool()]

    monkeypatch.setattr(settings, "MCP_RESEARCH_SERVERS", {
        "good": {"transport": "stdio", "command": "fake"},
        "broken": {"transport": "stdio", "command": "fake"},
    })
    monkeypatch.setattr(settings, "MCP_RESEARCH_ALLOWED_TOOLS", {
        "good": ["search_remote"], "broken": ["search_remote"],
    })
    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    owner = review.LiteratureReviewAgent()
    await owner.load()
    assert [tool.name for tool in owner.mcp_tools] == ["good__search_remote"]
    assert owner.mcp_status["broken"].startswith("unavailable")
    assert owner.get_graph() is not None


@pytest.mark.asyncio
async def test_mcp_papers_join_evidence(monkeypatch):
    class Tool:
        name = "remote__search_papers"
        args = {"query": {"type": "string"}}

        async def ainvoke(self, args):
            assert args["query"]
            return json.dumps({"papers": [{
                "title": "Remote Study", "abstract": "Remote result",
                "url": "https://example.org/paper",
            }]})

    owner = review.LiteratureReviewAgent()
    owner.mcp_tools = [Tool()]
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: json.dumps({"papers": []}))
    monkeypatch.setattr(review, "database_search_func", lambda query: "")
    monkeypatch.setattr(review, "normalize_search_query", lambda query: query)
    result = await review._search({"query": "retrieval"}, owner)
    assert result["evidence"][0]["source"] == "remote__search_papers"


@pytest.mark.asyncio
async def test_index_requires_approval_and_downloaded_path(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    monkeypatch.setattr(review, "DOWNLOAD_PAPERS_DIR", str(tmp_path))
    monkeypatch.setattr(review, "PDF_PATH_RE", re.compile(re.escape(str(pdf))))
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: json.dumps({"papers": []}))
    monkeypatch.setattr(review, "database_search_func", lambda query: "")
    calls = []
    monkeypatch.setattr(review, "create_vector_db_from_pdf_func", lambda path: calls.append(path) or "indexed")
    owner = review.LiteratureReviewAgent()
    graph = review._build_graph(owner)
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "index"}}
    await graph.ainvoke({"messages": [HumanMessage(content=f"请索引 {pdf}")]}, config)
    assert not calls
    await graph.ainvoke(Command(resume="approve"), config)
    assert calls == [str(pdf)]


@pytest.mark.asyncio
async def test_download_then_index_asks_twice(tmp_path, monkeypatch):
    from agents.tools import openreview

    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    monkeypatch.setattr(review, "DOWNLOAD_PAPERS_DIR", str(tmp_path))
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: json.dumps({"papers": []}))
    monkeypatch.setattr(review, "database_search_func", lambda query: "")
    monkeypatch.setattr(openreview, "download_paper_from_arxiv_func", lambda arxiv_id: json.dumps({
        "success": True, "file_path": str(pdf),
    }))
    indexed = []
    monkeypatch.setattr(review, "create_vector_db_from_pdf_func", lambda path: indexed.append(path) or "indexed")
    graph = review._build_graph(review.LiteratureReviewAgent())
    graph.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "combined"}}
    await graph.ainvoke({"messages": [HumanMessage(content="下载 arXiv:1234.5678 并建库")]}, config)
    assert (await graph.aget_state(config)).tasks[0].interrupts[0].value["action"]["kind"] == "download_arxiv"
    await graph.ainvoke(Command(resume="approve"), config)
    assert (await graph.aget_state(config)).tasks[0].interrupts[0].value["action"]["target"] == str(pdf)
    assert not indexed
    await graph.ainvoke(Command(resume="approve"), config)
    assert indexed == [str(pdf)]


def test_openreview_download_is_gated_and_missing_target_is_explained():
    state = {"messages": [HumanMessage(content="下载 OpenReview ID=abc_123 并综述")]}
    prepared = review._prepare(state)
    assert prepared["actions"] == [{"kind": "download_openreview", "target": "abc_123"}]
    missing = review._prepare({"messages": [HumanMessage(content="下载并建库")]})
    assert not missing["actions"]
    assert len(missing["action_results"]) == 2


@pytest.mark.asyncio
async def test_local_stdio_mcp_server_exposes_read_only_tools():
    client = MultiServerMCPClient({
        "scholar": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "service.mcp_server"],
            "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "backend/app")},
        }
    })
    tools = await client.get_tools()
    names = {tool.name for tool in tools}
    assert {"search_arxiv", "search_openreview", "list_downloaded_papers", "search_local_papers"} <= names
    assert "download_paper" not in names


def test_default_chroma_loader_returns_retriever(tmp_path, monkeypatch):
    from agents.tools import utils

    monkeypatch.setenv("VECTOR_DB_TYPE", "chroma")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(utils, "get_embeddings", lambda: FakeEmbeddings(size=4))
    retriever = utils.load_vector_db()
    assert retriever is not None
    assert retriever.invoke("anything") == []
