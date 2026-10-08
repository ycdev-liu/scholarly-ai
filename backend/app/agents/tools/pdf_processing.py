"""清理论文 PDF 页面文本，并为包含图注的页面生成预览。"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

_REFERENCES = re.compile(r"(?im)^\s*(?:references|bibliography|参考文献)\s*$")
_FIGURE = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*\d+")
_FIGURE_CAPTION = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*\d+\s*:")
_APPENDIX = re.compile(r"(?i)^(?:appendix\b|supplementary\b)")


def _starts_new_section(text: str) -> bool:
    """判断参考文献后是否进入附录等新章节。"""
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
    """识别图页解析后产生的大量单词碎片行。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 30 and sum(len(line.split()) <= 1 for line in lines) / len(lines) > 0.6


def _render_page_preview(pdf_path: str, page_index: int, output_path: Path) -> None:
    """渲染指定 PDF 页面，供界面展示和人工核验。"""
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
    """加载可检索的正文页，跳过参考文献并保留图页预览。"""
    pages = PyPDFLoader(pdf_path).load()
    documents: list[Document] = []
    in_references = False

    for page in pages:
        page_index = int(page.metadata["page"])
        text = page.page_content

        if in_references:
            # 参考文献可能跨页；遇到附录等新章节才恢复正文收录。
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
            # 优先保留图注；无图注时改用布局模式重新抽取页面文字。
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
            # 预览路径随页码元数据保存，问答结果可回溯到原图页。
            preview = preview_dir / f"page-{page_index + 1:03d}.png"
            _render_page_preview(pdf_path, page_index, preview)
            metadata["preview_path"] = str(preview)

        documents.append(Document(page_content=text.strip(), metadata=metadata))

    return documents
