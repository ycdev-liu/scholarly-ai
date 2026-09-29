"""Read-only paper tools for local MCP clients. Run with `python -m service.mcp_server`."""

from agents.tools.openreview import (
    list_downloaded_papers_func,
    openreview_search_func,
    search_arxiv_func,
)
from agents.tools.vector_db import database_search_func
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scholarly-ai")


@mcp.tool()
def search_arxiv(query: str, max_results: int = 8) -> str:
    """Search arXiv papers by topic."""
    return search_arxiv_func(query=query, max_results=min(max_results, 8))


@mcp.tool()
def search_openreview(keyword: str, max_papers: int = 8) -> str:
    """Search OpenReview papers by keyword."""
    return openreview_search_func(keyword=keyword, max_papers=min(max_papers, 8))


@mcp.tool()
def list_downloaded_papers() -> str:
    """List locally downloaded paper PDFs."""
    return list_downloaded_papers_func()


@mcp.tool()
def search_local_papers(query: str) -> str:
    """Search the local paper vector database."""
    return database_search_func(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
