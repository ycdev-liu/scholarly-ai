"""Auto 模式走现有 message_generator / SSE 的集成测试。工具全部 mock。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents import get_agent
from schema import StreamInput
from service import app
from service.utils import message_generator


@pytest.fixture(autouse=True)
def _stub_query_model():
    with patch(
        "agents.query_normalization._invoke_model",
        side_effect=RuntimeError("stream tests do not call the model"),
    ):
        yield

_MISS_DB = "No relevant documents found in the database for this query."
_EMPTY_PAPERS = json.dumps({"success": True, "total_files": 0, "papers": []})
_HIT_PAPERS = json.dumps(
    {
        "success": True,
        "total_files": 1,
        "papers": [{"filename": "医学图像分割.pdf", "path": "医学图像分割.pdf", "size_kb": 1}],
    }
)
_EXTERNAL = json.dumps(
    {
        "total_papers": 1,
        "papers": [
            {
                "title": "U-Net Segmentation",
                "openreview_url": "https://openreview.net/forum?id=abc",
            }
        ],
    }
)


def _events(chunks: list[str]) -> list[dict | str]:
    events: list[dict | str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            body = line.removeprefix("data: ")
            events.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return events


def _tool_names(events: list[dict | str]) -> list[str]:
    names: list[str] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        content = event["content"]
        for call in content.get("tool_calls") or []:
            names.append(call["name"])
    return names


async def _collect(query: str, thread_id: str) -> list[dict | str]:
    chunks: list[str] = []
    async for item in message_generator(
        StreamInput(message=query, thread_id=thread_id, stream_tokens=True),
        "auto",
    ):
        chunks.append(item)
    return _events(chunks)


@pytest_asyncio.fixture
async def sqlite_checkpointer(tmp_path):
    agent = get_agent("auto")
    previous = agent.checkpointer
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "auto.db")) as saver:
        await saver.setup()
        agent.checkpointer = saver
        yield saver
        agent.checkpointer = previous


@pytest.mark.asyncio
async def test_stream_case_a_sufficient_skips_external(sqlite_checkpointer):
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_HIT_PAPERS),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch("agents.auto_supervisor.search_arxiv_func") as search,
    ):
        events = await _collect("医学图像分割", "thread-sufficient")

    assert search.call_count == 0
    assert _tool_names(events) == ["List_Downloaded_Papers", "Database_Search"]
    finals = [
        event["content"]["content"]
        for event in events
        if isinstance(event, dict)
        and event["type"] == "message"
        and event["content"]["type"] == "ai"
        and event["content"]["content"]
    ]
    assert any("本地已有足够相关的论文" in text for text in finals)
    assert events[-1] == "[DONE]"
    assert all(event.get("type") != "error" for event in events if isinstance(event, dict))
    assert all(event["content"]["run_id"] for event in events if isinstance(event, dict))


@pytest.mark.asyncio
async def test_stream_case_b_miss_reaches_external_and_reuses_thread(sqlite_checkpointer):
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_EMPTY_PAPERS),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=_EXTERNAL) as search,
    ):
        events = await _collect("帮我找一些医学图像分割相关的论文", "thread-miss")
        second = await _collect("再看一次", "thread-miss")

    assert search.call_count >= 2
    assert _tool_names(events) == [
        "List_Downloaded_Papers",
        "Database_Search",
        "Search_ArXiv",
    ]
    tool_payloads = [
        event["content"]["content"]
        for event in events
        if isinstance(event, dict) and event["type"] == "message" and event["content"]["type"] == "tool"
    ]
    assert any("U-Net Segmentation" in text or "total_papers" in text for text in tool_payloads)
    finals = [
        event["content"]["content"]
        for event in events
        if isinstance(event, dict)
        and event["type"] == "message"
        and event["content"]["type"] == "ai"
        and not event["content"]["tool_calls"]
    ]
    assert any("U-Net Segmentation" in text for text in finals)
    assert events[-1] == "[DONE]"
    assert second[-1] == "[DONE]"
    assert all(event.get("type") != "error" for event in second if isinstance(event, dict))


def test_http_auto_stream_uses_existing_route(tmp_path, monkeypatch):
    from core import settings

    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "http.db"))
    body = {"message": "帮我找一些医学图像分割相关的论文", "thread_id": "http-thread"}
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_EMPTY_PAPERS),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=_EXTERNAL) as search,
        TestClient(app) as client,
    ):
        with client.stream("POST", "/api/agents/auto/stream", json=body) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            raw = "".join(response.iter_text())

    events = _events([raw])
    assert search.call_count >= 1
    assert "Search_ArXiv" in _tool_names(events)
    assert any(
        isinstance(event, dict)
        and event["type"] == "message"
        and "U-Net Segmentation" in event["content"]["content"]
        for event in events
    )
    assert events[-1] == "[DONE]"
