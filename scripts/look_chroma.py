from pathlib import Path
import chromadb

db = Path("data/vector_databases/attention_trial_20260929")
assert (db / "chroma.sqlite3").is_file(), f"数据库不存在：{db}"

client = chromadb.PersistentClient(path=str(db))
for collection in client.list_collections():
    print("集合:", collection.name, "片段数:", collection.count())
    rows = collection.get(limit=3, include=["documents", "metadatas", "embeddings"])
    for i, chunk_id in enumerate(rows["ids"]):
        meta = rows["metadatas"][i]
        print("\nID:", chunk_id)
        print("PDF:", meta.get("source"))
        print("页码:", meta.get("page", 0) + 1)
        print("图页:", meta.get("preview_path"))
        print("向量维度:", len(rows["embeddings"][i]))
        print("文本:", rows["documents"][i][:500])