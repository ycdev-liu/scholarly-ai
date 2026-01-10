"""
Agent 工具包
统一导出所有工具，方便 agent 调用
"""
from agents.tools.math_tools import calculator
from agents.tools.vector_db_tools import (
    database_search,
    create_vector_db_from_pdf,
    get_vector_db_info,
    switch_vector_db,
)
from agents.tools.paper_tools import (
    openreview_search,
    download_paper,
    download_paper_from_arxiv,
    list_downloaded_papers,
)
from agents.tools.utils import (
    get_embeddings,
    load_vector_db,
    format_contexts,
    clear_retriever_cache,
    DATA_BASE_DIR,
    VECTOR_DB_BASE_DIR,
    DOWNLOAD_BASE_DIR,
    DOWNLOAD_PAPERS_DIR,
)

# 导出所有工具
__all__ = [
    # 数学工具
    "calculator",
    # 向量数据库工具
    "database_search",
    "create_vector_db_from_pdf",
    "get_vector_db_info",
    "switch_vector_db",
    # 论文工具
    "openreview_search",
    "download_paper",
    "download_paper_from_arxiv",
    "list_downloaded_papers",
    # 工具函数
    "get_embeddings",
    "load_vector_db",
    "format_contexts",
    "clear_retriever_cache",
    # 常量
    "DATA_BASE_DIR",
    "VECTOR_DB_BASE_DIR",
    "DOWNLOAD_BASE_DIR",
    "DOWNLOAD_PAPERS_DIR",
]

# 工具分类
MATH_TOOLS = [calculator]
VECTOR_DB_TOOLS = [
    database_search,
    create_vector_db_from_pdf,
    get_vector_db_info,
    switch_vector_db,
]
PAPER_TOOLS = [
    openreview_search,
    download_paper,
    download_paper_from_arxiv,
    list_downloaded_papers,
]

# 所有工具的列表
ALL_TOOLS = MATH_TOOLS + VECTOR_DB_TOOLS + PAPER_TOOLS

