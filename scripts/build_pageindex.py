"""为已有知识库中的 PDF 补建 PageIndex 文档树。"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "app"))

from agents.tools.pageindex_search import index_pdf  # noqa: E402


def main() -> None:
    """读取知识库路径和 PDF 列表，逐篇建树并打印文档 ID。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="Existing Chroma or Qdrant knowledge base path")
    parser.add_argument("pdfs", nargs="+", help="PDF files to add")
    parser.add_argument("--collection", help="Qdrant collection name (omit for Chroma)")
    args = parser.parse_args()
    load_dotenv()
    for pdf in args.pdfs:
        result = index_pdf(args.db_path, pdf, args.collection)
        print(f"{pdf}: {result['doc_id']}")


if __name__ == "__main__":
    main()
