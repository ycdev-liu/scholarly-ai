"""验证前端知识库列表只展示可识别的本地库。"""
import asyncio

from service.routers.vectordb import list_vector_dbs


def test_list_vector_dbs_detects_chroma_and_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "vector_databases" / "papers"
    (db / "sources").mkdir(parents=True)
    (db / "chroma.sqlite3").touch()
    (db / "bm25.sqlite3").touch()
    (db / "sources" / "paper.pdf").touch()
    (tmp_path / "vector_databases" / "unfinished").mkdir()

    result = asyncio.run(list_vector_dbs())

    assert result["items"] == [{
        "name": "papers",
        "db_type": "chroma",
        "db_path": "vector_databases/papers",
        "collection_name": None,
        "source_count": 1,
        "has_bm25": True,
    }]


def test_list_vector_dbs_deduplicates_docker_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "data" / "vector_databases" / "papers"
    db.mkdir(parents=True)
    (db / "chroma.sqlite3").touch()
    (tmp_path / "vector_databases").symlink_to(tmp_path / "data" / "vector_databases", target_is_directory=True)

    result = asyncio.run(list_vector_dbs())

    assert len(result["items"]) == 1
