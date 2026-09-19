"""外部论文结果排序：去重、过滤无关项、保留 Top-K。不访问网络。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.auto_supervisor import auto_supervisor
from agents.result_ranking import rank_search_results

_QUERY = "semi-supervised fetal ultrasound segmentation"


def _arxiv(arxiv_id: str, title: str, abstract: str = "") -> dict:
    return {"title": title, "abstract": abstract, "arxiv_id": arxiv_id, "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"}


def _openreview(paper_id: str, title: str, abstract: str = "") -> dict:
    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "openreview_url": f"https://openreview.net/forum?id={paper_id}",
    }


def test_high_relevance_ranks_before_low_relevance():
    papers = [
        _arxiv("2", "A Survey of Image Segmentation", "general segmentation methods for natural images"),
        _arxiv(
            "3",
            "Semi-Supervised Fetal Ultrasound Image Segmentation",
            "semi-supervised segmentation of fetal ultrasound images",
        ),
    ]
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert [paper["arxiv_id"] for paper in ranked] == ["3", "2"]


def test_irrelevant_papers_are_removed():
    papers = [
        _arxiv("1", "Weather Forecasting with Transformers", "rainfall and climate prediction"),
        _arxiv("3", "Semi-Supervised Fetal Ultrasound Segmentation", "fetal ultrasound segmentation"),
    ]
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert [paper["arxiv_id"] for paper in ranked] == ["3"]


def test_duplicate_papers_are_collapsed():
    papers = [
        _arxiv("2301.00001v1", "Semi-Supervised Fetal Ultrasound Segmentation", "fetal"),
        _arxiv("2301.00001v2", "Semi-Supervised Fetal Ultrasound Segmentation", "fetal ultrasound semi-supervised segmentation"),
        {
            "title": "Semi-supervised fetal ultrasound segmentation",
            "abstract": "fetal ultrasound segmentation",
        },
        {
            "title": "  Semi-Supervised   Fetal Ultrasound Segmentation ",
            "abstract": "another copy without an id",
        },
    ]
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert len(ranked) == 2
    assert ranked[0]["arxiv_id"] == "2301.00001v2"


def test_empty_candidates():
    assert rank_search_results(_QUERY, [], top_k=5) == []
    assert rank_search_results(_QUERY, None, top_k=5) == []


def test_fewer_results_than_top_k():
    papers = [
        _arxiv("3", "Fetal Ultrasound Segmentation", "fetal ultrasound"),
        _arxiv("1", "Stock Price Prediction", "financial markets"),
    ]
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert len(ranked) == 1
    assert ranked[0]["arxiv_id"] == "3"


def test_openreview_shape_dedupes_on_id_and_filters():
    papers = [
        _openreview("abc", "Semi-Supervised Fetal Ultrasound Segmentation", "fetal ultrasound segmentation"),
        _openreview("abc", "Duplicate note", "fetal ultrasound"),
        _openreview("xyz", "Stock Market Prediction", "equity returns"),
    ]
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert [paper["id"] for paper in ranked] == ["abc"]
    assert "openreview_url" in ranked[0]


def test_top_k_truncates_relevant_papers():
    papers = [
        _arxiv(str(index), f"Fetal Ultrasound Segmentation Variant {index}", "fetal ultrasound segmentation")
        for index in range(6)
    ]
    papers[0]["title"] = "Semi-Supervised Fetal Ultrasound Segmentation"
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert len(ranked) == 5
    assert ranked[0]["arxiv_id"] == "0"


@pytest.mark.asyncio
async def test_final_answer_keeps_only_ranked_papers():
    raw = json.dumps(
        {
            "total_papers": 2,
            "papers": [
                _arxiv("1", "Stock Price Prediction", "financial markets"),
                _arxiv("3", "Semi-Supervised Fetal Ultrasound Segmentation", "fetal ultrasound segmentation"),
            ],
        }
    )
    with (
        patch("agents.auto_supervisor.list_downloaded_papers_func", return_value=json.dumps({"success": True, "total_files": 0, "papers": []})),
        patch("agents.auto_supervisor.database_search_func", return_value="No relevant documents found in the database for this query."),
        patch("agents.query_normalization._invoke_model", return_value=_QUERY),
        patch("agents.auto_supervisor.search_arxiv_func", return_value=raw),
    ):
        result = await auto_supervisor.ainvoke(
            {"messages": [HumanMessage(content="帮我找一些胎儿超声半监督分割方面的论文")]}
        )

    answer = result["messages"][-1].content
    assert "Semi-Supervised Fetal Ultrasound Segmentation" in answer
    assert "Stock Price Prediction" not in answer
