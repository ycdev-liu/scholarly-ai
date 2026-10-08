"""用人工标注的 PDF 页面评测证据检索效果。

默认只使用现有 Chroma 片段离线评测 BM25；传入 --online 后再评测向量检索、
RRF 混合检索，以及已建树知识库的 PageIndex。结果输出为 JSON，便于对照复现。
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "app"))

from agents.tools.hybrid_search import BM25Index, _existing_documents, fuse_results  # noqa: E402
from agents.tools.pageindex_search import _load_manifest, retrieve_pages  # noqa: E402


def score(questions, results):
    """计算逐题页面召回、首个相关页的倒数排名和检索耗时。"""
    recalls = []
    hits = []
    reciprocal_ranks = []
    latencies = []
    evidence_chars = []
    for question, (documents, elapsed_ms) in zip(questions, results):
        pages = []
        ranks = []
        gold = set(question["pages"])
        for rank, doc in enumerate(documents, 1):
            start = doc.metadata.get("page_start", doc.metadata.get("page"))
            if start is None:
                continue
            end = doc.metadata.get("page_end", start)
            covered = set(range(int(start), int(end) + 1))
            pages.extend(covered)
            if covered & gold:
                ranks.append(rank)
        hits.append(bool(ranks))
        # 同一页面可能对应多个片段，召回计算时按页去重。
        recalls.append(len(set(pages) & gold) / len(gold))
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
        latencies.append(elapsed_ms)
        evidence_chars.append(sum(len(doc.page_content) for doc in documents))
    sorted_ms = sorted(latencies)
    # 使用最近秩法计算 P95，小样本下结果只供冒烟验证。
    p95 = sorted_ms[max(0, math.ceil(0.95 * len(sorted_ms)) - 1)]
    return {
        "questions": len(questions),
        "page_recall": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "latency_p95_ms": round(p95, 2),
        "mean_evidence_chars": round(statistics.mean(evidence_chars)),
        "hit_ids": [q["id"] for q, hit in zip(questions, hits) if hit],
    }


def main():
    """加载知识库和标注题集，逐题执行可用的检索策略。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("tests/fixtures/attention_retrieval.json"))
    parser.add_argument("--online", action="store_true", help="Call configured embedding/PageIndex models")
    args = parser.parse_args()
    load_dotenv(".env")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    store = Chroma(persist_directory=str(args.db_path))
    bm25 = BM25Index(args.db_path)
    if bm25.is_empty():
        # 旧库可能只有向量索引，首次评测时从原片段补建词项索引。
        bm25.add_documents(_existing_documents(store, "chroma"))
    measurements = {"bm25": []}
    dense_retriever = None
    has_pageindex = bool(_load_manifest(args.db_path))
    if args.online:
        # 在线模式会调用配置的 Embedding/LLM 服务，需保证密钥可用。
        from agents.tools.utils import get_embeddings

        store._embedding_function = get_embeddings()
        dense_retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 40})
        measurements.update({"dense": [], "hybrid_rrf": []})
        if has_pageindex:
            measurements["hybrid_rrf_pageindex"] = []
    for q in questions:
        query = q["query"]
        start = time.perf_counter()
        lexical = bm25.search(query, limit=5)
        measurements["bm25"].append((lexical, (time.perf_counter() - start) * 1000))
        if dense_retriever is None:
            continue
        start = time.perf_counter()
        dense = dense_retriever.invoke(query)
        dense_ms = (time.perf_counter() - start) * 1000
        measurements["dense"].append((dense[:5], dense_ms))
        start = time.perf_counter()
        lexical_20 = bm25.search(query, limit=20)
        hybrid = fuse_results(dense, lexical_20, limit=5)
        hybrid_ms = dense_ms + (time.perf_counter() - start) * 1000
        measurements["hybrid_rrf"].append((hybrid, hybrid_ms))
        if has_pageindex:
            # 与片段级结果一起计入证据；两组的上下文预算不同，报告中予以注明。
            start = time.perf_counter()
            pages = retrieve_pages(query, args.db_path, dense + lexical_20)
            measurements["hybrid_rrf_pageindex"].append(
                (hybrid + pages, hybrid_ms + (time.perf_counter() - start) * 1000)
            )
    report = {
        "corpus": str(args.db_path),
        "gold_unit": "zero-based PDF page; page_recall averages each question's relevant-page coverage",
        "budgets": "5 chunks for BM25/dense/RRF; RRF+PageIndex adds at most 2 pages",
        "methods": {name: score(questions, rows) for name, rows in measurements.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
