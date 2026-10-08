import io

import pytest
from agents.tools.hybrid_search import BM25Index, HybridRetriever, fuse_results
from langchain_core.documents import Document


def _doc(text: str, source: str = "paper.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 0})


def test_bm25_finds_exact_terms_chinese_and_identifiers(tmp_path):
    index = BM25Index(tmp_path / "papers")
    target = _doc("PagedAttention reduces KV cache fragmentation", "2309.06180.pdf")
    other = _doc("A study of unrelated weather models", "weather.pdf")
    index.add_documents([target, other])

    assert index.search("PagedAttention")[0].page_content == target.page_content
    assert index.search("2309.06180")[0].page_content == target.page_content
    assert index.search('PagedAttention" OR unrelated')[0].page_content == target.page_content

    index.add_documents([_doc("医学图像分割方法", "medicine.pdf")])
    assert index.search("图像分割")[0].metadata["source"] == "medicine.pdf"


def test_bm25_is_scoped_and_persistent(tmp_path):
    first = BM25Index(tmp_path / "first")
    second = BM25Index(tmp_path / "second")
    first.add_documents([_doc("PagedAttention")])
    second.add_documents([_doc("Transformer")])

    assert BM25Index(tmp_path / "first").search("PagedAttention")
    assert second.search("PagedAttention") == []


def test_rrf_promotes_result_found_by_both_retrievers():
    dense_only = _doc("semantic match")
    lexical_only = _doc("keyword match")
    common = _doc("both kinds of match")

    results = fuse_results([dense_only, common], [lexical_only, common])

    assert [doc.page_content for doc in results] == [
        "both kinds of match",
        "semantic match",
        "keyword match",
    ]


def test_hybrid_backfills_old_chroma_index_and_recovers_dense_miss(tmp_path):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    path = tmp_path / "old_chroma"
    store = Chroma(embedding_function=FakeEmbeddings(size=4), persist_directory=str(path))
    store.add_documents([_doc("PagedAttention manages memory")])

    class DenseMiss:
        def invoke(self, query):
            return []

    retriever = HybridRetriever(DenseMiss(), store, "chroma", path)
    assert retriever.invoke("PagedAttention")[0].page_content == "PagedAttention manages memory"
    assert BM25Index(path).search("PagedAttention")


def test_hybrid_backfills_old_qdrant_collection(tmp_path):
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models

    path = tmp_path / "old_qdrant"
    client = QdrantClient(path=str(path))
    client.create_collection(
        "papers",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name="papers",
        embedding=FakeEmbeddings(size=4),
    )
    store.add_documents([_doc("PagedAttention manages memory")])

    class DenseMiss:
        def invoke(self, query):
            return []

    retriever = HybridRetriever(DenseMiss(), store, "qdrant", path)
    assert retriever.invoke("PagedAttention")[0].page_content == "PagedAttention manages memory"
    assert BM25Index(path, "papers").search("PagedAttention")
    client.close()


def test_pdf_indexing_and_database_search_use_both_indexes(tmp_path, monkeypatch):
    import json

    from agents.tools import utils, vector_db
    from langchain_core.embeddings import FakeEmbeddings

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fake pdf; loader is replaced")
    base_dir = tmp_path / "indexes"
    embeddings = FakeEmbeddings(size=4)
    monkeypatch.setattr(vector_db, "VECTOR_DB_BASE_DIR", str(base_dir))
    monkeypatch.setattr(vector_db, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(utils, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(
        vector_db,
        "load_paper_pages",
        lambda *_: [_doc("PagedAttention reduces KV cache fragmentation")],
    )
    monkeypatch.setenv("VECTOR_DB_TYPE", "chroma")

    result = json.loads(vector_db.create_vector_db_from_pdf_func(str(pdf), db_name="paper"))
    assert result["success"] is True
    indexed = BM25Index(result["db_path"]).search("PagedAttention")
    assert indexed[0].metadata["chunking_strategy"] == "section_paragraph_token"
    assert indexed[0].metadata["page_start"] == indexed[0].metadata["page_end"] == 0
    assert "PagedAttention" in vector_db.database_search_func("PagedAttention")
    utils.clear_retriever_cache()


def test_qdrant_pdf_indexing_and_database_search_use_both_indexes(tmp_path, monkeypatch):
    import json

    from agents.tools import utils, vector_db
    from langchain_core.embeddings import FakeEmbeddings

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fake pdf; loader is replaced")
    embeddings = FakeEmbeddings(size=4)
    monkeypatch.setattr(vector_db, "VECTOR_DB_BASE_DIR", str(tmp_path / "indexes"))
    monkeypatch.setattr(vector_db, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(utils, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(
        vector_db,
        "load_paper_pages",
        lambda *_: [_doc("PagedAttention reduces KV cache fragmentation")],
    )
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.delenv("QDRANT_URL", raising=False)

    result = json.loads(
        vector_db.create_vector_db_from_pdf_func(str(pdf), db_name="paper", db_type="qdrant")
    )
    assert result["success"] is True, result
    assert BM25Index(result["db_path"], "documents").search("PagedAttention")
    assert "PagedAttention" in vector_db.database_search_func("PagedAttention")
    utils.clear_retriever_cache()


@pytest.mark.asyncio
async def test_document_upload_builds_searchable_chroma_and_bm25(tmp_path, monkeypatch):
    from fastapi import UploadFile
    from langchain_core.embeddings import FakeEmbeddings
    from service import document_processing
    from service.routers.document import upload_and_process_documents

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(document_processing, "get_embeddings", lambda **_: FakeEmbeddings(size=4))
    upload = UploadFile(file=io.BytesIO(b"PagedAttention manages the KV cache."), filename="paper.txt")

    result = await upload_and_process_documents(
        files=[upload],
        chunk_size=100,
        chunk_overlap=0,
        use_local_embedding=False,
        model_name="unused",
        db_name="upload",
        db_type="chroma",
        auto_switch=False,
    )

    assert result["success"] is True, result["errors"]
    assert BM25Index(result["db_path"]).search("PagedAttention")
