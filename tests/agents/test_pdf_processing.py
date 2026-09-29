from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agents.tools import pdf_processing
from agents.tools.utils import format_contexts


def _page(number: int, text: str) -> Document:
    return Document(page_content=text, metadata={"page": number, "source": "paper.pdf", "total_pages": 4})


def test_references_are_excluded_but_later_sections_are_kept(monkeypatch):
    pages = [
        _page(0, "Main result.\nReferences\n[1] Citation"),
        _page(1, "[2] Another citation"),
        _page(2, "Appendix A\nAdditional derivation."),
    ]
    monkeypatch.setattr(
        pdf_processing, "PyPDFLoader", lambda _: SimpleNamespace(load=lambda: pages)
    )

    documents = pdf_processing.load_paper_pages("paper.pdf")

    assert [doc.metadata["page"] for doc in documents] == [0, 2]
    assert documents[0].page_content == "Main result."
    assert documents[1].page_content == "Appendix A\nAdditional derivation."


def test_noisy_visual_page_keeps_caption_and_preview(monkeypatch, tmp_path):
    text = "Attention Visualizations\n" + "word\n" * 40 + "Figure 3: Attention heads align.\n13"
    pages = [_page(0, text)]
    monkeypatch.setattr(
        pdf_processing, "PyPDFLoader", lambda _: SimpleNamespace(load=lambda: pages)
    )
    rendered = []
    monkeypatch.setattr(
        pdf_processing, "_render_page_preview", lambda *args: rendered.append(args)
    )

    documents = pdf_processing.load_paper_pages("paper.pdf", tmp_path)

    assert documents[0].page_content.startswith("Attention Visualizations\nFigure 3:")
    assert "word\nword" not in documents[0].page_content
    assert documents[0].metadata["preview_path"] == str(tmp_path / "page-001.png")
    assert rendered == [("paper.pdf", 0, Path(tmp_path / "page-001.png"))]


def test_retrieved_context_includes_source_page_and_preview():
    doc = Document(
        page_content="Attention heads align.",
        metadata={
            "source": "/papers/paper.pdf",
            "page": 3,
            "preview_path": "previews/page-004.png",
        },
    )

    assert format_contexts([doc]) == (
        "Source: paper.pdf, page 4\n"
        "Attention heads align.\n"
        "Page preview: previews/page-004.png"
    )
