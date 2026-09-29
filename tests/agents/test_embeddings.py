from types import SimpleNamespace

import pytest
from agents.tools import utils
from service import document_processing


def test_dashscope_embedding_provider_is_shared_by_indexing_and_search(monkeypatch):
    created = []

    def fake_dashscope_embeddings(**kwargs):
        embedding = SimpleNamespace(**kwargs)
        created.append(embedding)
        return embedding

    monkeypatch.setenv("EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    monkeypatch.setattr("langchain_community.embeddings.DashScopeEmbeddings", fake_dashscope_embeddings)
    monkeypatch.setattr(utils, "_embeddings_cache", None)

    search_embeddings = utils.get_embeddings()
    upload_embeddings = document_processing.get_embeddings(use_local=True)

    assert search_embeddings is upload_embeddings
    assert len(created) == 1
    assert created[0].model == "text-embedding-v3"
    assert created[0].dashscope_api_key == "test-key"


def test_dashscope_embedding_provider_requires_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(utils, "_embeddings_cache", None)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        utils.get_embeddings()
