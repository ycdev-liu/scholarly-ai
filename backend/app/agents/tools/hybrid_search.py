"""将持久化 BM25 词项检索与现有向量检索融合。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .pageindex_search import pageindex_enabled, retrieve_pages

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
_RRF_K = 60
_CANDIDATES = 20
_RESULTS = 5


def _terms(text: str) -> list[str]:
    """英文按标识符拆词，中文使用相邻双字，供 FTS5 建索引和查询。"""
    result: list[str] = []
    for word in _WORD.findall(text.lower()):
        if "\u4e00" <= word[0] <= "\u9fff":
            result.extend(word[i : i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                result.append(word)
        else:
            parts = re.findall(r"[a-z0-9]+", word)
            result.extend(parts)
            if len(parts) > 1:
                result.append("".join(parts))
    return result


def chunk_id(doc: Document) -> str:
    """为同一片段生成稳定 ID，以便去重并合并两路检索结果。"""
    existing = doc.metadata.get("chunk_id")
    if existing:
        return str(existing)
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    payload = f"{source}\0{page}\0{doc.page_content}".encode()
    return hashlib.sha256(payload).hexdigest()


class BM25Index:
    """每个知识库对应一个 SQLite FTS5 索引；Qdrant 集合另加命名空间。"""

    def __init__(self, db_path: str | Path, namespace: str | None = None):
        suffix = re.sub(r"[^a-zA-Z0-9_-]", "_", namespace) if namespace else ""
        filename = f"bm25-{suffix}.sqlite3" if suffix else "bm25.sqlite3"
        self.path = Path(db_path) / filename

    def _connect(self) -> sqlite3.Connection:
        """打开词项索引，首次访问时创建 FTS5 表。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "chunk_id UNINDEXED, content UNINDEXED, metadata UNINDEXED, tokens)"
        )
        return connection

    def is_empty(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0

    def add_documents(self, documents: Iterable[Document]) -> None:
        """按片段 ID 更新索引，重复入库不会产生重复记录。"""
        with self._connect() as connection:
            for doc in documents:
                identity = chunk_id(doc)
                metadata = dict(doc.metadata)
                metadata["chunk_id"] = identity
                source = Path(str(metadata.get("source", ""))).name
                tokens = " ".join(_terms(f"{source} {doc.page_content}"))
                connection.execute("DELETE FROM chunks WHERE chunk_id = ?", (identity,))
                connection.execute(
                    "INSERT INTO chunks(chunk_id, content, metadata, tokens) VALUES (?, ?, ?, ?)",
                    (identity, doc.page_content, json.dumps(metadata, ensure_ascii=False, default=str), tokens),
                )

    def search(self, query: str, limit: int = _CANDIDATES) -> list[Document]:
        """使用 FTS5 的 BM25 排序返回原文片段及其元数据。"""
        terms = list(dict.fromkeys(_terms(query)))
        if not terms:
            return []
        # 将词项逐一加引号，避免用户输入被当作 FTS 查询语法执行。
        expression = " OR ".join('"' + term + '"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT content, metadata FROM chunks WHERE tokens MATCH ? "
                "ORDER BY bm25(chunks) LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [Document(page_content=content, metadata=json.loads(metadata)) for content, metadata in rows]


def _existing_documents(vectorstore: Any, db_type: str) -> Iterable[Document]:
    """分批读取旧向量库中的片段，为尚无 BM25 的知识库补建索引。"""
    if db_type == "chroma":
        offset = 0
        while True:
            batch = vectorstore.get(include=["documents", "metadatas"], limit=500, offset=offset)
            texts = batch.get("documents") or []
            metadata = batch.get("metadatas") or []
            if not texts:
                break
            yield from (Document(page_content=text, metadata=meta or {}) for text, meta in zip(texts, metadata))
            offset += len(texts)
    else:
        client = vectorstore.client
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=vectorstore.collection_name,
                offset=offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                text = payload.get(vectorstore.content_payload_key, "")
                metadata = payload.get(vectorstore.metadata_payload_key, {})
                if text:
                    yield Document(page_content=text, metadata=metadata or {})
            if offset is None:
                break


def fuse_results(dense: list[Document], lexical: list[Document], limit: int = _RESULTS) -> list[Document]:
    """用倒数排名融合（RRF）合并两路结果，避免直接比较不同量纲的分数。"""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in (dense, lexical):
        seen: set[str] = set()
        for rank, doc in enumerate(ranking, start=1):
            identity = chunk_id(doc)
            if identity in seen:
                continue
            seen.add(identity)
            # 同一片段命中两路时累加排名贡献，每一路只计一次。
            scores[identity] = scores.get(identity, 0.0) + 1 / (_RRF_K + rank)
            documents.setdefault(identity, doc)
    ordered = sorted(scores, key=lambda identity: -scores[identity])
    return [documents[identity] for identity in ordered[:limit]]


class HybridRetriever:
    """先融合片段候选，再按需追加 PageIndex 定位的原文页面。"""

    def __init__(self, dense_retriever: Any, vectorstore: Any, db_type: str, db_path: str | Path):
        self.dense_retriever = dense_retriever
        self.vectorstore = vectorstore
        self.db_path = db_path
        namespace = vectorstore.collection_name if db_type == "qdrant" else None
        self.namespace = namespace
        self.lexical = BM25Index(db_path, namespace)
        if self.lexical.is_empty():
            # 兼容启用混合检索之前创建的向量库。
            self.lexical.add_documents(_existing_documents(vectorstore, db_type))

    def invoke(self, query: str) -> list[Document]:
        """返回最多 5 个融合片段，启用 PageIndex 时再追加引用页面。"""
        dense = self.dense_retriever.invoke(query)
        lexical = self.lexical.search(query)
        chunks = fuse_results(dense, lexical)
        if not pageindex_enabled():
            return chunks
        try:
            pages = retrieve_pages(query, self.db_path, dense + lexical, self.namespace)
        except Exception:
            # 页面定位失败时仍返回传统 RAG 的片段证据。
            logger.exception("PageIndex retrieval failed; using chunk retrieval")
            return chunks
        return chunks + pages
