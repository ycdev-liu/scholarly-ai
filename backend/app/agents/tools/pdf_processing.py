"""Prepare paper PDF pages for text retrieval and visual inspection."""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


_REFERENCES = re.compile(r"(?im)^\s*(?:references|bibliography|参考文献)\s*$")
_FIGURE = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*\d+")
_FIGURE_CAPTION = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*\d+\s*:")
_APPENDIX = re.compile(r"(?i)^(?:appendix\b|supplementary\b)")


def _starts_new_section(text: str) -> bool:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(
        _APPENDIX.match(first)
        or (
            1 < len(first.split()) <= 5
            and len(first) < 80
            and first.istitle()
            and not first.endswith((".", ",", ":"))
        )
    )


def _noisy_figure_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 30 and sum(len(line.split()) <= 1 for line in lines) / len(lines) > 0.6


def _render_page_preview(pdf_path: str, page_index: int, output_path: Path) -> None:
    import pypdfium2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pypdfium2.PdfDocument(pdf_path) as pdf:
        page = pdf[page_index]
        try:
            bitmap = page.render(scale=1.5)
            try:
                bitmap.to_pil().save(output_path)
            finally:
                bitmap.close()
        finally:
            page.close()


def load_paper_pages(pdf_path: str, preview_dir: Path | None = None) -> list[Document]:
    """Load searchable pages, omitting the bibliography and preserving figure previews."""
    pages = PyPDFLoader(pdf_path).load()
    documents: list[Document] = []
    in_references = False

    for page in pages:
        page_index = int(page.metadata["page"])
        text = page.page_content

        if in_references:
            if not _starts_new_section(text):
                continue
            in_references = False

        match = _REFERENCES.search(text)
        if match:
            text = text[: match.start()]
            in_references = True

        if not text.strip():
            continue

        if _noisy_figure_text(text):
            captions = list(_FIGURE_CAPTION.finditer(text))
            if captions:
                heading = text.splitlines()[0].strip()
                text = text[captions[-1].start():].strip()
                if heading.istitle() and len(heading) < 80:
                    text = f"{heading}\n{text}"
            else:
                from pypdf import PdfReader

                text = PdfReader(pdf_path).pages[page_index].extract_text(
                    extraction_mode="layout"
                ) or ""

        if not text.strip():
            continue

        metadata = dict(page.metadata)
        if preview_dir is not None and _FIGURE.search(text):
            preview = preview_dir / f"page-{page_index + 1:03d}.png"
            _render_page_preview(pdf_path, page_index, preview)
            metadata["preview_path"] = str(preview)

        documents.append(Document(page_content=text.strip(), metadata=metadata))

    return documents
