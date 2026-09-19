"""Auto Supervisor 的分类与路由测试。不访问真实向量库或 OpenReview。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.auto_supervisor import (
    auto_supervisor,
    evaluate_local_results,
    requires_fresh_search,
    select_external_source,
    to_arxiv_query,
)


@pytest.fixture(autouse=True)
def _stub_query_model():
    """路由测试不调用真实模型。失败会落到已有短语 fallback。"""
    with patch(
        "agents.query_normalization._invoke_model",
        side_effect=RuntimeError("routing tests do not call the model"),
    ):
        yield

_MISS_DB = "No relevant documents found in the database for this query."


def _papers(*names: str) -> str:
    return json.dumps(
        {
            "success": True,
            "total_files": len(names),
            "papers": [{"filename": name, "path": name, "size_kb": 1} for name in names],
        }
    )


def test_auto_is_default_and_old_agents_remain():
    from agents.agents import DEFAULT_AGENT, agents, get_agent, get_all_agent_info
    from agents.auto_supervisor import auto_supervisor

    assert DEFAULT_AGENT == "auto"
    assert get_agent("auto") is auto_supervisor
    keys = [item.key for item in get_all_agent_info()]
    assert keys == [
        "auto",
        "rag-assistant",
        "openreview-agent",
        "paper-research-supervisor",
    ]
    assert keys.index(DEFAULT_AGENT) == 0
    for agent_id in ("rag-assistant", "openreview-agent", "paper-research-supervisor"):
        assert get_agent(agent_id) is agents[agent_id].graph_like


def test_evaluate_local_results_failed_when_database_search_errors():
    error = (
        "Database search error: We couldn't connect to 'https://huggingface.co' "
        "to load the files, and couldn't find them in the cached files."
    )
    assert evaluate_local_results("医学图像分割", _papers(), error) == "FAILED"
    assert evaluate_local_results("医学图像分割", _papers("医学图像分割.pdf"), error) == "FAILED"


@pytest.mark.asyncio
async def test_graph_database_failure_still_calls_arxiv():
    external = json.dumps(
        {
            "total_papers": 1,
            "papers": [{"title": "U-Net Segmentation", "pdf_url": "https://arxiv.org/pdf/1505.04597"}],
        }
    )
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_papers()),
        patch(
            "agents.auto_supervisor.database_search_func",
            return_value="Database search error: embedding model failed to load",
        ),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=external) as search,
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="帮我找一些医学图像分割相关的论文")]}
        )

    search.assert_called()
    assert result["local_status"] == "FAILED"
    assert result["used_external"] is True
    assert any(
        getattr(message, "content", "") == "Local Search → FAILED → fallback to external search"
        for message in result["messages"]
    )
    assert "U-Net Segmentation" in result["messages"][-1].content


def test_evaluate_local_results_miss_when_nothing_local():
    assert evaluate_local_results("医学图像分割", _papers(), _MISS_DB) == "MISS"


def test_evaluate_local_results_insufficient_when_unrelated():
    assert (
        evaluate_local_results("医学图像分割", _papers("attention_transformer.pdf"), "weather policy")
        == "INSUFFICIENT"
    )


def test_evaluate_local_results_sufficient_on_filename_or_chunk():
    assert (
        evaluate_local_results("医学图像分割", _papers("医学图像分割综述.pdf"), _MISS_DB)
        == "SUFFICIENT"
    )
    assert (
        evaluate_local_results("segmentation", _papers(), "a paper about image segmentation")
        == "SUFFICIENT"
    )


def test_select_external_source_prefers_arxiv_unless_openreview():
    assert select_external_source("帮我找一些医学图像分割相关的论文") == "arxiv"
    assert select_external_source("OpenReview submission about segmentation") == "openreview"
    assert select_external_source("这篇投稿的 conference review 意见") == "openreview"
    assert to_arxiv_query("帮我找一些医学图像分割相关的论文") == "medical image segmentation"
    assert requires_fresh_search("找一些 2026 年最新的论文")
    assert requires_fresh_search("latest medical image papers")
    assert not requires_fresh_search("帮我找一些医学图像分割相关的论文")


@pytest.mark.asyncio
async def test_graph_local_miss_continues_to_openreview():
    external = json.dumps(
        {
            "total_papers": 1,
            "papers": [{"title": "U-Net Segmentation", "openreview_url": "https://openreview.net/forum?id=abc"}],
        }
    )
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_papers()),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=external) as search,
        patch("agents.auto_supervisor.openreview_search_func") as openreview,
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="帮我找一些医学图像分割相关的论文")]}
        )

    search.assert_called()
    openreview.assert_not_called()
    assert result["local_status"] == "MISS"
    assert result["used_external"] is True
    assert "U-Net Segmentation" in result["messages"][-1].content
    tool_names = [message.name for message in result["messages"] if isinstance(message, ToolMessage)]
    assert tool_names == ["List_Downloaded_Papers", "Database_Search", "Search_ArXiv"]


@pytest.mark.asyncio
async def test_graph_insufficient_also_continues_external():
    with (
        patch(
            "agents.auto_supervisor.list_downloaded_papers_func",
            return_value=_papers("unrelated_notes.pdf"),
        ),
        patch("agents.auto_supervisor.database_search_func", return_value="company handbook"),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=json.dumps({"total_papers": 0, "papers": []})) as search,
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="医学图像分割")]}
        )

    search.assert_called()
    assert result["local_status"] == "INSUFFICIENT"
    assert "没有找到足够相关的论文" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_sufficient_does_not_call_external():
    with (
        patch(
            "agents.auto_supervisor.list_downloaded_papers_func",
            return_value=_papers("医学图像分割.pdf"),
        ),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch("agents.auto_supervisor.search_arxiv_func") as search,
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="医学图像分割")]}
        )

    search.assert_not_called()
    assert result["local_status"] == "SUFFICIENT"
    assert result["used_external"] is False
    assert "本地已有足够相关的论文" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_fresh_query_skips_local_database():
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func") as listed,
        patch("agents.auto_supervisor.database_search_func") as searched,
        patch(
            "agents.auto_supervisor.search_arxiv_func",
            return_value=json.dumps(
                {"total_papers": 1, "papers": [{"title": "Medical Image Segmentation 2026", "pdf_url": ""}]}
            ),
        ) as external,
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="最新的医学图像分割论文")]}
        )

    listed.assert_not_called()
    searched.assert_not_called()
    external.assert_called()
    assert result["skip_local"] is True
    assert "Medical Image Segmentation 2026" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_external_search_uses_normalized_query():
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=_papers()),
        patch("agents.auto_supervisor.database_search_func", return_value=_MISS_DB),
        patch(
            "agents.query_normalization._invoke_model",
            return_value="semi-supervised fetal ultrasound segmentation",
        ),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=json.dumps({"total_papers": 0, "papers": []})) as search,
    ):
        await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="帮我找一些胎儿超声半监督分割方面的论文")]}
        )

    assert search.call_args_list[0].kwargs == {
        "query": 'all:"semi-supervised fetal ultrasound segmentation"',
        "max_results": 8,
    }
