"""使用 PageIndex 文档树定位页面，与片段级 RAG 检索配合。"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_MAX_DOCUMENTS = 3
_MAX_PAGES = 2


def pageindex_enabled() -> bool:
    """从环境变量读取 PageIndex 开关，默认关闭。"""
    return os.getenv("PAGEINDEX_ENABLED", "false").lower() in {"1", "true", "yes"}


def _namespace_path(db_path: str | Path, namespace: str | None) -> Path:
    """为 Qdrant 集合隔离 PageIndex 文件，Chroma 直接使用知识库目录。"""
    if not namespace:
        return Path(db_path)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", namespace)
    return Path(db_path) / f"pageindex-{safe}"


def _client(db_path: str | Path, namespace: str | None = None) -> Any:
    """创建本地 PageIndex 客户端，可覆盖建树和查树模型的连接参数。"""
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("PageIndex 未安装：运行 uv sync --extra pageindex") from exc

    index_model = os.getenv("PAGEINDEX_INDEX_MODEL", "gpt-4.1-mini")
    chat_model = os.getenv("PAGEINDEX_CHAT_MODEL", "gpt-4.1")
    backend = {}
    if key := os.getenv("PAGEINDEX_LLM_API_KEY"):
        backend["api_key"] = key
    if base_url := os.getenv("PAGEINDEX_LLM_BASE_URL"):
        backend["base_url"] = base_url
    index = {"model": index_model, "storage_path": str(_namespace_path(db_path, namespace) / "index")}
    chat = {"model": chat_model}
    if backend:
        index["backend"] = backend
        chat["backend"] = backend
    return PageIndexClient(
        index=index,
        chat=chat,
    )


def _manifest_path(db_path: str | Path, namespace: str | None = None) -> Path:
    """返回 PDF 来源路径与 PageIndex 文档 ID 的映射文件路径。"""
    return _namespace_path(db_path, namespace) / "pageindex-documents.json"


def _load_manifest(db_path: str | Path, namespace: str | None = None) -> list[dict[str, str]]:
    """读取当前知识库已建树的 PDF 列表。"""
    path = _manifest_path(db_path, namespace)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid PageIndex manifest: {path}")
    return data


def index_pdf(
    db_path: str | Path, pdf_path: str | Path, namespace: str | None = None
) -> dict[str, str]:
    """为 PDF 建立文档树，并将来源路径和文档 ID 持久化。"""
    source = Path(pdf_path).resolve()
    if source.suffix.lower() != ".pdf" or not source.is_file():
        raise ValueError(f"PageIndex 需要可读取的 PDF 文件: {source}")
    manifest = _load_manifest(db_path, namespace)
    for item in manifest:
        if item.get("source") == str(source):
            # 已建树的 PDF 直接复用原文档 ID。
            return item

    client = _client(db_path, namespace)
    result = client.submit_document(str(source), wait=True)
    doc_id = result.get("doc_id")
    if not doc_id:
        raise RuntimeError("PageIndex did not return a document ID")
    item = {"source": str(source), "doc_id": str(doc_id)}
    manifest.append(item)
    path = _manifest_path(db_path, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # 先写临时文件再替换，避免进程中断留下半份映射文件。
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return item


def retrieve_pages(
    query: str,
    db_path: str | Path,
    candidate_chunks: list[Document],
    namespace: str | None = None,
) -> list[Document]:
    """在候选论文中查找引用页，只返回该页原文及页码元数据。"""
    manifest = _load_manifest(db_path, namespace)
    if not manifest:
        return []

    candidate_sources = {str(doc.metadata.get("source", "")) for doc in candidate_chunks}
    # 优先按完整来源路径匹配；历史数据路径不同则退化为文件名匹配。
    selected = [item for item in manifest if item["source"] in candidate_sources]
    if not selected:
        candidate_names = {Path(source).name for source in candidate_sources}
        selected = [item for item in manifest if Path(item["source"]).name in candidate_names]
    if not selected and len(manifest) <= _MAX_DOCUMENTS:
        # 小知识库无来源匹配时仍尝试已建树论文，避免完全丢失页面证据。
        selected = manifest
    selected = selected[:_MAX_DOCUMENTS]
    if not selected:
        return []

    client = _client(db_path, namespace)
    ids = [item["doc_id"] for item in selected]
    answer = client.chat(query, doc_id=ids, citations=True)
    # 生成答案仅用于提取引用；下游问答只接收 get_page_content 读取的原文。
    citations = client.get_citations(answer, doc_id=ids)
    sources = {item["doc_id"]: item["source"] for item in selected}
    pages: list[Document] = []
    seen: set[tuple[str, int]] = set()
    for citation in citations:
        doc_id = str(citation.get("doc_id", ""))
        try:
            page_number = int(citation.get("page"))
        except (TypeError, ValueError):
            continue
        if doc_id not in sources or page_number < 1 or (doc_id, page_number) in seen:
            continue
        seen.add((doc_id, page_number))
        content = client.get_page_content(doc_id, str(page_number))
        for page in content:
            text = str(page.get("markdown") or "").strip()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": sources[doc_id],
                            # LangChain 的 PDF 页码从 0 开始，PageIndex 引用页码从 1 开始。
                            "page": page_number - 1,
                            "retrieval_method": "pageindex",
                            "pageindex_doc_id": doc_id,
                        },
                    )
                )
                break
        if len(pages) >= _MAX_PAGES:
            break
    logger.info("PageIndex selected %s pages from %s documents", len(pages), len(selected))
    return pages
