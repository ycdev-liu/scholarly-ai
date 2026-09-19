"""外部检索结果的去重、相关性排序和 Top-K。不调用模型。"""

from __future__ import annotations

import re

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "using",
    "based",
    "via",
    "paper",
    "papers",
    "study",
    "method",
    "methods",
    "approach",
    "towards",
    "novel",
    "new",
    "recent",
    "latest",
    "their",
    "our",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "not",
    "but",
    "can",
    "may",
    "also",
    "such",
    "than",
    "then",
    "over",
    "under",
    "about",
    "its",
    "you",
    "your",
    "which",
    "what",
    "when",
    "where",
    "how",
    "why",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "by",
    "or",
    "as",
    "at",
    "is",
    "be",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]{2,}")
_EMPTY_TITLES = {"", "n/a", "na", "untitled", "none", "null"}


def _norm_title(title: str | None) -> str:
    text = (title or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _arxiv_key(paper: dict) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip().lower()
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id


def _dedup_key(paper: dict, index: int) -> str:
    arxiv_id = _arxiv_key(paper)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    stable_id = str(paper.get("id") or "").strip().lower()
    if stable_id and stable_id not in _EMPTY_TITLES:
        return f"id:{stable_id}"
    title = _norm_title(str(paper.get("title") or ""))
    if title and title not in _EMPTY_TITLES:
        return f"title:{title}"
    return f"row:{index}"


def dedupe_papers(papers: list[dict]) -> list[dict]:
    """同一篇只留一条。有 arxiv_id 用它，否则用稳定 id，再否则用规范化标题。"""
    chosen: dict[str, dict] = {}
    order: list[str] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            continue
        key = _dedup_key(paper, index)
        current = chosen.get(key)
        if current is None:
            chosen[key] = paper
            order.append(key)
            continue
        if len(str(paper.get("abstract") or "")) > len(str(current.get("abstract") or "")):
            chosen[key] = paper
    return [chosen[key] for key in order]


def _terms(text: str) -> set[str]:
    found: set[str] = set()
    for token in _TOKEN_RE.findall((text or "").lower()):
        if token in _STOP or len(token) < 3:
            continue
        found.add(token)
        if "-" in token:
            found.add(token.replace("-", ""))
            found.update(part for part in token.split("-") if len(part) >= 3 and part not in _STOP)
    return found


def _score(query_terms: set[str], paper: dict) -> int:
    if not query_terms:
        return 0
    title_terms = _terms(str(paper.get("title") or ""))
    abstract_terms = _terms(str(paper.get("abstract") or ""))
    score = 0
    for term in query_terms:
        if term in title_terms:
            score += 3
        elif term in abstract_terms:
            score += 1
    return score


def rank_search_results(query: str, papers: list[dict] | None, top_k: int = 5) -> list[dict]:
    """去重后按标题/摘要与 query 的词重叠排序，丢掉明显无关项，保留 Top-K。"""
    if not papers or top_k <= 0:
        return []
    unique = dedupe_papers(papers)
    query_terms = _terms(query)
    scored = [(index, _score(query_terms, paper), paper) for index, paper in enumerate(unique)]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [paper for _, _, paper in scored[:top_k]]
