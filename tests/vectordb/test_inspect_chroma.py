import importlib.util
from pathlib import Path

import chromadb

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_chroma.py"
spec = importlib.util.spec_from_file_location("inspect_chroma", SCRIPT)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


def test_inspector_reads_chunks_and_fields_without_embedding_queries(tmp_path):
    db_path = tmp_path / "paper_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection("papers")
    collection.add(
        ids=["first", "second"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["First chunk", "Second chunk"],
        metadatas=[
            {"source": "paper.pdf", "page": 0},
            {"source": "paper.pdf", "page": 1, "preview_path": "previews/page-002.png"},
        ],
    )

    assert inspector.find_databases(tmp_path) == [db_path]
    chunks = inspector.load_chunks(collection)
    assert [chunk["text"] for chunk in chunks] == ["First chunk", "Second chunk"]
    assert [inspector.page_number(chunk["metadata"]) for chunk in chunks] == [1, 2]

    fields = inspector.field_summary(chunks).set_index("字段")
    assert fields.loc["source", "片段数"] == 2
    assert fields.loc["preview_path", "片段数"] == 1


def test_page_number_accepts_missing_or_invalid_metadata():
    assert inspector.page_number({}) is None
    assert inspector.page_number({"page": "bad"}) is None
