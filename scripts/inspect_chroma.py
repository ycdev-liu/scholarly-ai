"""Read-only Chroma chunk inspector.

Run from the project root:
    uv run streamlit run scripts/inspect_chroma.py --server.address 127.0.0.1 --server.port 8503
"""

from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "vector_databases"
BATCH_SIZE = 500


def find_databases(root: Path = DB_ROOT) -> list[Path]:
    """Only offer existing Chroma stores; PersistentClient creates missing ones."""
    if not root.is_dir():
        return []
    return sorted(path.parent for path in root.rglob("chroma.sqlite3"))


def load_chunks(collection) -> list[dict]:
    """Read stored text and metadata without fetching every embedding."""
    chunks = []
    for offset in range(0, collection.count(), BATCH_SIZE):
        result = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            chunks.append({"id": chunk_id, "text": text or "", "metadata": metadata or {}})
    return chunks


def page_number(metadata: dict) -> int | None:
    try:
        return int(metadata["page"]) + 1
    except (KeyError, TypeError, ValueError):
        return None


def field_summary(chunks: list[dict]) -> pd.DataFrame:
    counts = Counter(key for chunk in chunks for key in chunk["metadata"])
    rows = []
    for name, count in sorted(counts.items()):
        sample = next(
            (chunk["metadata"][name] for chunk in chunks if name in chunk["metadata"]),
            "",
        )
        rows.append({"字段": name, "片段数": count, "示例值": str(sample)[:120]})
    return pd.DataFrame(rows, columns=["字段", "片段数", "示例值"])


def _active_database_index(paths: list[Path]) -> int:
    configured = dotenv_values(PROJECT_ROOT / ".env").get("CHROMA_DB_PATH")
    if configured:
        active = (PROJECT_ROOT / str(configured)).resolve()
        if active in paths:
            return paths.index(active)
    return 0


def main() -> None:
    st.set_page_config(page_title="向量库片段检查", layout="wide", initial_sidebar_state="auto")
    st.markdown("### 向量库片段检查")

    paths = find_databases()
    if not paths:
        st.warning(f"未找到 Chroma 数据库：{DB_ROOT}")
        st.stop()

    with st.sidebar:
        st.markdown("#### 数据源")
        db_path = st.selectbox(
            "数据库",
            paths,
            index=_active_database_index(paths),
            format_func=lambda path: path.name,
        )

    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    if not collections:
        st.info("数据库中没有集合。")
        st.stop()

    with st.sidebar:
        collection_name = st.selectbox("集合", [item.name for item in collections])

    collection = client.get_collection(collection_name)
    chunks = load_chunks(collection)
    if not chunks:
        st.info("集合中没有片段。")
        st.stop()

    sources = sorted({str(chunk["metadata"].get("source", "")) for chunk in chunks})
    with st.sidebar:
        st.markdown("#### 筛选")
        source = st.selectbox(
            "论文",
            [None, *sources],
            format_func=lambda value: "全部" if value is None else Path(value).name,
        )
        page_options = sorted(
            {number for chunk in chunks if (number := page_number(chunk["metadata"])) is not None}
        )
        page = st.selectbox(
            "页码",
            [None, *page_options],
            format_func=lambda value: "全部" if value is None else str(value),
        )
        keyword = st.text_input("文本包含").strip().casefold()
        with_preview = st.toggle("仅含图页", value=False)

    filtered = [
        chunk
        for chunk in chunks
        if (source is None or chunk["metadata"].get("source", "") == source)
        and (page is None or page_number(chunk["metadata"]) == page)
        and (not keyword or keyword in chunk["text"].casefold())
        and (not with_preview or bool(chunk["metadata"].get("preview_path")))
    ]

    sources_count = len({chunk["metadata"].get("source", "") for chunk in chunks})
    page_count = len(
        {
            (chunk["metadata"].get("source", ""), number)
            for chunk in chunks
            if (number := page_number(chunk["metadata"])) is not None
        }
    )
    st.caption(
        f"全部片段 {len(chunks)} · 当前结果 {len(filtered)} · "
        f"论文 {sources_count} · 索引页数 {page_count}"
    )
    st.caption(str(db_path))
    if not filtered:
        st.info("没有匹配的片段。")
        st.stop()

    table = pd.DataFrame(
        {
            "列表序号": index + 1,
            "页码": page_number(chunk["metadata"]),
            "片段预览": chunk["text"].replace("\n", " ")[:160],
            "字符数": len(chunk["text"]),
            "图页": "有" if chunk["metadata"].get("preview_path") else "",
            "论文": Path(str(chunk["metadata"].get("source", ""))).name,
        }
        for index, chunk in enumerate(filtered)
    )
    selection = st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=380,
        on_select="rerun",
        selection_mode="single-row",
        key=f"chunks-{db_path.name}-{collection_name}-{source}-{page}-{keyword}-{with_preview}",
        column_config={
            "片段预览": st.column_config.TextColumn("片段预览", width="large"),
            "论文": st.column_config.TextColumn("论文", width="medium"),
        },
    )
    selected_rows = selection.selection.rows
    selected = filtered[selected_rows[0]] if selected_rows else filtered[0]
    metadata = selected["metadata"]

    st.markdown("#### 片段详情")
    st.caption(f"ID {selected['id']}")
    text_tab, fields_tab, vector_tab, preview_tab = st.tabs(
        ["切分文本", "存储字段", "向量", "图页"]
    )
    with text_tab:
        st.code(selected["text"], language=None, wrap_lines=True)

    with fields_tab:
        st.json({"id": selected["id"], "metadata": metadata}, expanded=True)
        st.dataframe(field_summary(chunks), hide_index=True, use_container_width=True)

    with vector_tab:
        result = collection.get(ids=[selected["id"]], include=["embeddings"])
        embeddings = result["embeddings"]
        if embeddings is None or len(embeddings) == 0:
            st.info("该片段没有存储向量。")
        else:
            vector = np.asarray(embeddings[0], dtype=float)
            stats = st.columns(4)
            for column, label, value in zip(
                stats,
                ("维度", "L2 范数", "最小值", "最大值"),
                (
                    len(vector),
                    f"{np.linalg.norm(vector):.4f}",
                    f"{vector.min():.4f}",
                    f"{vector.max():.4f}",
                ),
            ):
                column.metric(label, value)
            st.line_chart(pd.DataFrame({"数值": vector[:64]}), height=220)
            st.dataframe(
                pd.DataFrame({"维度索引": range(min(32, len(vector))), "数值": vector[:32]}),
                hide_index=True,
                use_container_width=True,
            )

    with preview_tab:
        preview_path = metadata.get("preview_path")
        preview = PROJECT_ROOT / str(preview_path) if preview_path else None
        if preview is not None and preview.is_file():
            st.image(str(preview), use_container_width=True)
            st.download_button(
                "下载图页",
                data=preview.read_bytes(),
                file_name=preview.name,
                mime="image/png",
                icon=":material/download:",
            )
        else:
            st.info("该片段没有关联图页。")


if __name__ == "__main__":
    main()
