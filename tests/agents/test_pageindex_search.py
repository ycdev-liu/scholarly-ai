import io
import json
import sys
from types import SimpleNamespace

import pytest
from agents.tools import hybrid_search, pageindex_search
from langchain_core.documents import Document


def _doc(source: str, text: str = "chunk") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 0})


def test_pageindex_client_supports_openai_compatible_backend(tmp_path, monkeypatch):
    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "pageindex", SimpleNamespace(PageIndexClient=client))
    monkeypatch.setenv("PAGEINDEX_INDEX_MODEL", "openai/qwen-plus")
    monkeypatch.setenv("PAGEINDEX_CHAT_MODEL", "openai/qwen-plus")
    monkeypatch.setenv("PAGEINDEX_LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAGEINDEX_LLM_BASE_URL", "https://example.com/v1")
    pageindex_search._client(tmp_path)

    assert captured["index"]["backend"] == {
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
    }
    assert captured["chat"]["backend"] == captured["index"]["backend"]


def test_index_pdf_persists_document_id_and_is_idempotent(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"test")
    submitted = []
    client = SimpleNamespace(
        submit_document=lambda path, wait: submitted.append((path, wait)) or {"doc_id": "pi-1"}
    )
    monkeypatch.setattr(pageindex_search, "_client", lambda *_: client)

    first = pageindex_search.index_pdf(tmp_path / "db", pdf, "papers")
    second = pageindex_search.index_pdf(tmp_path / "db", pdf, "papers")

    assert first == second == {"source": str(pdf), "doc_id": "pi-1"}
    assert submitted == [(str(pdf), True)]
    manifest = tmp_path / "db" / "pageindex-papers" / "pageindex-documents.json"
    assert json.loads(manifest.read_text()) == [first]


def test_pageindex_returns_cited_original_pages_only(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"test")
    other = tmp_path / "other.pdf"
    other.write_bytes(b"test")
    db_path = tmp_path / "db"
    db_path.mkdir()
    pageindex_search._manifest_path(db_path).write_text(
        json.dumps(
            [
                {"source": str(pdf), "doc_id": "pi-1"},
                {"source": str(other), "doc_id": "pi-2"},
            ]
        )
    )

    class FakeClient:
        def chat(self, query, *, doc_id, citations):
            assert query == "What is the result?"
            assert doc_id == ["pi-1"]
            assert citations is True
            return 'The answer is invented. <cite doc="paper.pdf" page="3"/>'

        def get_citations(self, answer, *, doc_id):
            assert doc_id == ["pi-1"]
            return [
                {"doc_id": "pi-1", "page": 3},
                {"doc_id": "pi-1", "page": 3},
                {"doc_id": "pi-2", "page": 1},
            ]

        def get_page_content(self, doc_id, pages):
            assert (doc_id, pages) == ("pi-1", "3")
            return [{"page_index": 3, "markdown": "Original experimental result."}]

    monkeypatch.setattr(pageindex_search, "_client", lambda *_: FakeClient())
    results = pageindex_search.retrieve_pages(
        "What is the result?", db_path, [_doc(str(pdf))]
    )

    assert len(results) == 1
    assert results[0].page_content == "Original experimental result."
    assert results[0].metadata["source"] == str(pdf)
    assert results[0].metadata["page"] == 2
    assert results[0].metadata["retrieval_method"] == "pageindex"


def test_hybrid_keeps_chunks_and_adds_pageindex_evidence(tmp_path, monkeypatch):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    path = tmp_path / "db"
    store = Chroma(embedding_function=FakeEmbeddings(size=4), persist_directory=str(path))
    store.add_documents([_doc("paper.pdf", "Chunk evidence")])
    retriever = hybrid_search.HybridRetriever(
        SimpleNamespace(invoke=lambda _: [_doc("paper.pdf", "Chunk evidence")]),
        store,
        "chroma",
        path,
    )
    monkeypatch.setattr(hybrid_search, "pageindex_enabled", lambda: True)
    monkeypatch.setattr(
        hybrid_search,
        "retrieve_pages",
        lambda *_: [_doc("paper.pdf", "Page evidence")],
    )

    results = retriever.invoke("question")

    assert [doc.page_content for doc in results] == ["Chunk evidence", "Page evidence"]


def test_pageindex_failure_preserves_traditional_retrieval(tmp_path, monkeypatch):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    path = tmp_path / "db"
    store = Chroma(embedding_function=FakeEmbeddings(size=4), persist_directory=str(path))
    store.add_documents([_doc("paper.pdf", "Chunk evidence")])
    retriever = hybrid_search.HybridRetriever(
        SimpleNamespace(invoke=lambda _: [_doc("paper.pdf", "Chunk evidence")]),
        store,
        "chroma",
        path,
    )
    monkeypatch.setattr(hybrid_search, "pageindex_enabled", lambda: True)

    def fail(*_):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(hybrid_search, "retrieve_pages", fail)
    assert retriever.invoke("question")[0].page_content == "Chunk evidence"


@pytest.mark.asyncio
async def test_pdf_upload_adds_pageindex_tree(tmp_path, monkeypatch):
    from fastapi import UploadFile
    from langchain_core.embeddings import FakeEmbeddings
    from service import document_processing
    from service.routers import document

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(document_processing, "get_embeddings", lambda **_: FakeEmbeddings(size=4))
    monkeypatch.setattr(document, "load_paper_pages", lambda *_: [_doc("paper.pdf", "Evidence in PDF")])
    monkeypatch.setattr(document, "pageindex_enabled", lambda: True)

    def fake_index(db_path, pdf_path, namespace):
        assert pdf_path.exists()
        assert pdf_path.name == "paper.pdf"
        assert namespace is None
        return {"doc_id": "pi-1"}

    monkeypatch.setattr(document, "index_pdf", fake_index)
    result = await document.upload_and_process_documents(
        files=[UploadFile(file=io.BytesIO(b"fake pdf"), filename="paper.pdf")],
        chunk_size=100,
        chunk_overlap=0,
        use_local_embedding=False,
        model_name="unused",
        db_name="upload",
        db_type="chroma",
        auto_switch=False,
    )
    assert result["success"] is True, result["errors"]
    assert result["pageindex"] == [{"filename": "paper.pdf", "doc_id": "pi-1", "success": True}]
    assert (tmp_path / "vector_databases" / "upload" / "sources" / "paper.pdf").read_bytes() == b"fake pdf"
    assert hybrid_search.BM25Index(result["db_path"]).search("Evidence")[0].metadata["chunking_strategy"] == "section_paragraph_token"
