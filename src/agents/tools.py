"""
工具模块 - 向后兼容层
所有工具已迁移到 agents.tools 包中，此文件用于保持向后兼容性
"""
# 从新的工具包中导入所有工具
from agents.tools import (
    # 数学工具
    calculator,
    # 向量数据库工具
    database_search,
    create_vector_db_from_pdf,
    get_vector_db_info,
    switch_vector_db,
    # 论文工具
    openreview_search,
    download_paper,
    download_paper_from_arxiv,
    list_downloaded_papers,
    # 工具函数
    get_embeddings,
    load_vector_db,
    format_contexts,
    clear_retriever_cache,
    # 常量
    DATA_BASE_DIR,
    VECTOR_DB_BASE_DIR,
    DOWNLOAD_BASE_DIR,
    DOWNLOAD_PAPERS_DIR,
)

# 导出所有内容以保持向后兼容
__all__ = [
    "calculator",
    "database_search",
    "create_vector_db_from_pdf",
    "get_vector_db_info",
    "switch_vector_db",
    "openreview_search",
    "download_paper",
    "download_paper_from_arxiv",
    "list_downloaded_papers",
    "get_embeddings",
    "load_vector_db",
    "format_contexts",
    "clear_retriever_cache",
    "DATA_BASE_DIR",
    "VECTOR_DB_BASE_DIR",
    "DOWNLOAD_BASE_DIR",
    "DOWNLOAD_PAPERS_DIR",
]
