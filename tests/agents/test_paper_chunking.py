import io

import pytest
from agents.tools.paper_chunking import split_paper_pages
from agents.tools.utils import format_contexts
from langchain_core.documents import Document


def _page(number: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source": "paper.pdf", "page": number})


def test_sections_continue_across_pages_but_do_not_mix_with_subsections():
    pages = [
        _page(
            0, "Abstract\nA compact summary.\n1 Introduction\nThe first page of the introduction."
        ),
        _page(
            1, "The introduction continues here.\n1.1 Motivation\nA separate motivation paragraph."
        ),
    ]

    chunks = split_paper_pages(pages, max_tokens=100, overlap_tokens=0)

    assert [chunk.metadata["section_path"] for chunk in chunks] == [
        "Abstract",
        "1 Introduction",
        "1 Introduction > 1.1 Motivation",
    ]
    assert (chunks[1].metadata["page_start"], chunks[1].metadata["page_end"]) == (0, 1)
    assert "pages 1-2" in format_contexts([chunks[1]])
    assert all(chunk.metadata["source"] == "paper.pdf" for chunk in chunks)


def test_long_paragraph_is_split_by_token_limit_without_losing_chinese_text():
    text = "科研论文分析方法" * 120
    chunks = split_paper_pages([_page(3, f"2 方法\n{text}")], max_tokens=48, overlap_tokens=8)

    assert len(chunks) > 1
    assert all(chunk.metadata["token_count"] <= 48 for chunk in chunks)
    assert all(chunk.metadata["page_start"] == chunk.metadata["page_end"] == 3 for chunk in chunks)
    assert all(chunk.metadata["section_path"] == "2 方法" for chunk in chunks)
    assert all("�" not in chunk.page_content for chunk in chunks)


def test_table_value_is_not_mistaken_for_heading():
    chunks = split_paper_pages(
        [
            _page(
                0,
                "3 Results\n28.4 BLEU on translation task.\nN d model dff h d k dv Pdrop\nThis is the result.",
            )
        ],
        max_tokens=100,
        overlap_tokens=0,
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["section_path"] == "3 Results"


def test_section_does_not_continue_across_removed_pages():
    chunks = split_paper_pages(
        [_page(1, "7 Conclusion\nMain conclusion."), _page(4, "Figure 3: Additional examples.")],
        max_tokens=100,
        overlap_tokens=0,
    )
    assert [chunk.metadata["section_path"] for chunk in chunks] == [
        "7 Conclusion",
        "Unclassified Section",
    ]


def test_chinese_section_and_subsection_headers():
    chunks = split_paper_pages(
        [_page(0, "一、研究方法\n本文介绍研究方法。\n（一）数据处理\n按段落切分论文内容。")],
        max_tokens=100,
        overlap_tokens=0,
    )
    assert [chunk.metadata["section_path"] for chunk in chunks] == [
        "一、研究方法",
        "一、研究方法 > （一）数据处理",
    ]


@pytest.mark.asyncio
async def test_pdf_without_text_reports_failure(tmp_path, monkeypatch):
    from fastapi import UploadFile
    from langchain_core.embeddings import FakeEmbeddings
    from service import document_processing
    from service.routers import document

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(document_processing, "get_embeddings", lambda **_: FakeEmbeddings(size=4))
    monkeypatch.setattr(document, "load_paper_pages", lambda *_: [])
    result = await document.upload_and_process_documents(
        files=[UploadFile(file=io.BytesIO(b"image-only PDF"), filename="scan.pdf")],
        db_name="scan",
        db_type="chroma",
        auto_switch=False,
        use_local_embedding=False,
        chunk_size=512,
        chunk_overlap=64,
    )

    assert result["success"] is False
    assert any("OCR" in error for error in result["errors"])
