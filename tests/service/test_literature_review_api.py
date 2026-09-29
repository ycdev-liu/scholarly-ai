import json
from unittest.mock import patch

from agents import literature_review as review
from agents.skills import discover_skills
from fastapi.testclient import TestClient
from langchain_community.chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from service import app


def test_review_approval_over_http_and_sse(monkeypatch):
    owner = review.LiteratureReviewAgent()
    owner.skills = discover_skills()
    monkeypatch.setattr(review, "search_arxiv_func", lambda query, max_results: json.dumps({"papers": []}))
    monkeypatch.setattr(review, "database_search_func", lambda query: "")
    monkeypatch.setattr(review, "get_model", lambda name: FakeListChatModel(responses=["unused"]))
    graph = review._build_graph(owner)
    graph.checkpointer = MemorySaver()
    with (
        patch("service.routers.agent.get_agent", return_value=graph),
        patch("service.utils.get_agent", return_value=graph),
    ):
        client = TestClient(app)
        request = {"message": "下载 arXiv:1234.5678 并综述", "thread_id": "http-review"}
        first = client.post("/api/agents/literature-review/invoke", json=request)
        assert first.status_code == 200, first.text
        assert first.json()["custom_data"]["kind"] == "approval_required"
        pending = client.get("/api/agents/literature-review/pending", params={"thread_id": "http-review"})
        assert pending.json()["pending"]["action"]["kind"] == "download_arxiv"

        missing = client.post("/api/agents/literature-review/invoke", json={"message": "", "thread_id": "http-review"})
        assert missing.status_code == 422

        denied = client.post("/api/agents/literature-review/invoke", json={"message": "", "thread_id": "http-review", "approval": "deny"})
        assert denied.status_code == 200, denied.text
        assert "取消" in denied.json()["content"]
        assert client.get(
            "/api/agents/literature-review/pending", params={"thread_id": "http-review"}
        ).json() == {"pending": None}

        second = client.post("/api/agents/literature-review/stream", json={"message": "下载 arXiv:1234.5678", "thread_id": "stream-review"})
        assert second.status_code == 200
        assert '"type": "approval_required"' in second.text
