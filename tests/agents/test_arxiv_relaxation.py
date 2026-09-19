"""arXiv 查询放宽：第一次够用就停止，不够再逐级 AND。不访问网络。"""

import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.arxiv_relaxation import arxiv_search_queries, collect_arxiv_candidates
from agents.result_ranking import rank_search_results

_QUERY = "fetal ultrasound semi-supervised segmentation"


def _paper(arxiv_id: str, title: str) -> dict:
    return {
        "title": title,
        "abstract": "semi-supervised fetal ultrasound segmentation",
        "arxiv_id": arxiv_id,
    }


def test_query_plan_keeps_topic_constraints():
    assert arxiv_search_queries(_QUERY) == [
        'all:"fetal ultrasound semi-supervised segmentation"',
        'all:"fetal ultrasound" AND all:"semi-supervised segmentation"',
        'all:"fetal ultrasound" AND all:"segmentation"',
        'all:"ultrasound" AND all:"semi-supervised segmentation"',
    ]


def test_first_search_with_enough_results_does_not_expand():
    calls: list[str] = []

    def search(expression: str, max_results: int = 8) -> list[dict]:
        calls.append(expression)
        return [_paper(str(index), f"Fetal Ultrasound Segmentation {index}") for index in range(5)]

    papers, attempts = collect_arxiv_candidates(_QUERY, search, min_candidates=5)
    assert calls == ['all:"fetal ultrasound semi-supervised segmentation"']
    assert len(papers) == 5
    assert attempts[0][1] == 5


def test_empty_first_search_uses_relaxed_query():
    calls: list[str] = []

    def search(expression: str, max_results: int = 8) -> list[dict]:
        calls.append(expression)
        if len(calls) == 1:
            return []
        return [_paper(str(index), f"Fetal Ultrasound Segmentation {index}") for index in range(5)]

    papers, attempts = collect_arxiv_candidates(_QUERY, search, min_candidates=5)
    assert attempts[0] == ('all:"fetal ultrasound semi-supervised segmentation"', 0)
    assert "AND" in attempts[1][0]
    assert attempts[1][1] == 5
    assert len(calls) == 2
    assert len(papers) == 5


def test_rounds_are_merged_and_duplicates_ranked_once():
    shared = _paper("2301.1", "Semi-Supervised Fetal Ultrasound Segmentation")
    extra = _paper("2301.2", "Fetal Ultrasound Image Segmentation")

    def search(expression: str, max_results: int = 8) -> list[dict]:
        if "AND" not in expression:
            return [shared]
        if expression.startswith('all:"fetal ultrasound" AND all:"semi-supervised'):
            return [shared, extra]
        return [_paper("2301.3", "Ultrasound Segmentation")]

    papers, attempts = collect_arxiv_candidates(_QUERY, search, min_candidates=2)
    assert [count for _, count in attempts] == [1, 2]
    assert len(papers) == 3
    ranked = rank_search_results(_QUERY, papers, top_k=5)
    assert [paper["arxiv_id"] for paper in ranked] == ["2301.1", "2301.2"]


def test_all_queries_can_return_nothing():
    calls: list[str] = []

    def search(expression: str, max_results: int = 8) -> list[dict]:
        calls.append(expression)
        return []

    papers, attempts = collect_arxiv_candidates(_QUERY, search, min_candidates=5)
    assert papers == []
    assert [count for _, count in attempts] == [0, 0, 0, 0]
    assert calls == arxiv_search_queries(_QUERY)


def test_stop_once_enough_candidates_are_found():
    calls: list[str] = []

    def search(expression: str, max_results: int = 8) -> list[dict]:
        calls.append(expression)
        if len(calls) == 1:
            return []
        return [_paper(str(index), f"Fetal Ultrasound Segmentation {index}") for index in range(5)]

    _, attempts = collect_arxiv_candidates(_QUERY, search, min_candidates=5)
    assert len(attempts) == 2
    assert len(calls) == 2
    assert attempts[-1][1] == 5
