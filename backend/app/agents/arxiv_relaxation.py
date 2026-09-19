"""arXiv 查询逐级放宽。只生成确定性的 search_query，不调用模型。"""

from __future__ import annotations

from agents.result_ranking import dedupe_papers


def _tokens(query: str) -> list[str]:
    return [part for part in (query or "").split() if part]


def _quoted(tokens: list[str]) -> str:
    return f'all:"{" ".join(tokens)}"'


def _and(left: list[str], right: list[str]) -> str:
    if not left or not right:
        return ""
    return f"{_quoted(left)} AND {_quoted(right)}"


def _topic_and_method(tokens: list[str]) -> tuple[list[str], list[str]]:
    """把检索词分成主题短语和方法短语。连字符词视为方法修饰。"""
    hyphen_at = next((index for index, token in enumerate(tokens) if "-" in token), None)
    if hyphen_at is None:
        if len(tokens) >= 3:
            return tokens[:-1], tokens[-1:]
        if len(tokens) == 2:
            return [tokens[0]], [tokens[1]]
        return tokens, []
    if hyphen_at == 0:
        topic = tokens[1:]
        method = [tokens[0]]
        if len(topic) >= 2:
            method = [tokens[0], topic[-1]]
            topic = topic[:-1]
        return topic, method
    return tokens[:hyphen_at], tokens[hyphen_at:]


def arxiv_search_queries(query: str) -> list[str]:
    """从严格短语到保持主题约束的 AND 组合。不会拆成单词 OR。"""
    tokens = _tokens(query)
    if not tokens:
        return []
    queries = [_quoted(tokens)]
    if len(tokens) == 1:
        return queries
    topic, method = _topic_and_method(tokens)
    candidates = [_and(topic, method)]
    if method and "-" in method[0] and len(method) > 1:
        candidates.append(_and(topic, method[1:]))
    if len(topic) > 1 and method:
        candidates.append(_and(topic[1:], method))
    seen = set(queries)
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            queries.append(item)
    return queries


def collect_arxiv_candidates(
    query: str,
    search,
    *,
    min_candidates: int = 5,
    max_results: int = 8,
) -> tuple[list[dict], list[tuple[str, int]]]:
    """按放宽顺序搜索。唯一候选达到 min_candidates 后停止。"""
    merged: list[dict] = []
    attempts: list[tuple[str, int]] = []
    for expression in arxiv_search_queries(query):
        try:
            found = search(expression, max_results) or []
        except Exception:
            found = []
        papers = [paper for paper in found if isinstance(paper, dict)]
        attempts.append((expression, len(papers)))
        merged.extend(papers)
        if len(dedupe_papers(merged)) >= min_candidates:
            break
    return merged, attempts
