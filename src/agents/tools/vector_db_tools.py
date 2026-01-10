"""
向量数据库工具模块
包含向量数据库的搜索、创建、切换等功能
"""
import json
import os
import re
import logging
import shutil
from datetime import datetime
from langchain_core.tools import BaseTool, tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from agents.tools.utils import (
    get_embeddings,
    format_contexts,
    clear_retriever_cache,
    _get_retriever,
    VECTOR_DB_BASE_DIR,
    DOWNLOAD_PAPERS_DIR,
)


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Starting database search for query: {query}")
        
        # 使用缓存的 retriever，避免每次都重新创建数据库连接
        retriever = _get_retriever()
        logger.debug("Retriever obtained successfully")
        
        # Search the database for relevant documents
        documents = retriever.invoke(query)
        logger.debug(f"Search completed, found {len(documents) if documents else 0} documents")
        
        if not documents:
            # 返回明确信息，告诉 Agent 搜索已完成但没有找到相关内容
            logger.warning(f"No documents found for query: {query}")
            return "No relevant documents found in the database for this query. The database search completed successfully, but no matching content was retrieved."
        
        # Format the documents into a string
        context_str = format_contexts(documents)
        logger.debug(f"Formatted context length: {len(context_str)} characters")
        
        return context_str
    except Exception as e:
        # 记录详细错误信息
        logger.error(f"Database search error for query '{query}': {e}", exc_info=True)
        # 返回错误信息，让 Agent 知道发生了什么
        error_msg = f"Database search error: {str(e)}"
        logger.error(f"Returning error message to agent: {error_msg}")
        return error_msg


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"


def create_vector_db_from_pdf_func(
    pdf_file_path: str,
    db_name: str = "",
    db_type: str = "",
    chunk_size: int = 2000,
    chunk_overlap: int = 500,
) -> str:
    """
    从PDF文件创建向量数据库。
    
    这个工具可以从下载的论文PDF文件创建向量数据库，使用 Qdrant。
    创建后，数据库可以被 Database_Search 工具使用。
    数据库默认保存在统一的数据目录下的 vector_databases 文件夹。
    
    Args:
        pdf_file_path: PDF文件的完整路径或文件名（例如：./data/downloads/papers/paper_xxx.pdf）
                      如果只提供文件名，会在统一目录 ./data/downloads/papers/ 中查找
                      如果文件名不存在，会尝试在目录中查找包含关键词的PDF文件
        db_name: 数据库名称（可选，如果不提供或为空则自动生成，基于时间戳）
        db_type: 数据库类型（可选，默认使用环境变量 VECTOR_DB_TYPE，默认为 "qdrant"）
        chunk_size: 文本块大小，默认 2000
        chunk_overlap: 文本块重叠大小，默认 500
    
    Returns:
        str: JSON字符串，包含创建结果信息
    """
    logger = logging.getLogger(__name__)
    
    try:
        original_path = pdf_file_path
        
        # 如果路径是相对路径且不包含目录，尝试在统一下载目录中查找
        if not os.path.isabs(pdf_file_path) and os.path.dirname(pdf_file_path) == "":
            # 只有文件名，在统一目录中查找
            pdf_file_path = os.path.join(DOWNLOAD_PAPERS_DIR, pdf_file_path)
        elif not os.path.isabs(pdf_file_path) and not pdf_file_path.startswith("./data"):
            # 如果是旧路径格式，尝试转换为新路径
            old_path = pdf_file_path
            if pdf_file_path.startswith("./downloads"):
                # 转换为新路径
                filename = os.path.basename(pdf_file_path)
                pdf_file_path = os.path.join(DOWNLOAD_PAPERS_DIR, filename)
                logger.info(f"检测到旧路径格式，尝试在新位置查找: {old_path} -> {pdf_file_path}")
        
        # 如果文件不存在，尝试在目录中查找包含关键词的文件
        if not os.path.exists(pdf_file_path):
            logger.warning(f"文件不存在: {pdf_file_path}，尝试在目录中查找...")
            
            # 提取关键词（从原始路径或文件名）
            search_keywords = []
            if original_path:
                # 从文件名提取关键词
                base_name = os.path.splitext(os.path.basename(original_path))[0].lower()
                # 移除常见的前缀和后缀
                base_name = base_name.replace("paper_", "").replace("arxiv_", "").replace("attention", "").replace("transformer", "")
                # 提取可能的 arXiv ID 或关键词
                arxiv_match = re.search(r'(\d+\.\d+)', base_name)
                if arxiv_match:
                    search_keywords.append(arxiv_match.group(1))
                # 添加其他关键词
                if "attention" in original_path.lower():
                    search_keywords.append("attention")
                if "transformer" in original_path.lower():
                    search_keywords.append("transformer")
            
            # 在目录中查找匹配的PDF文件
            if os.path.exists(DOWNLOAD_PAPERS_DIR):
                for file in os.listdir(DOWNLOAD_PAPERS_DIR):
                    if file.endswith('.pdf'):
                        file_lower = file.lower()
                        # 检查是否包含任何关键词
                        if any(keyword.lower() in file_lower for keyword in search_keywords if keyword):
                            found_path = os.path.join(DOWNLOAD_PAPERS_DIR, file)
                            logger.info(f"找到匹配的文件: {found_path}")
                            pdf_file_path = found_path
                            break
                        # 如果没有关键词，但文件名包含 "attention" 或 "transformer"，也尝试
                        elif ("attention" in file_lower and "attention" in original_path.lower()) or \
                             ("transformer" in file_lower and "transformer" in original_path.lower()):
                            found_path = os.path.join(DOWNLOAD_PAPERS_DIR, file)
                            logger.info(f"找到匹配的文件（基于关键词）: {found_path}")
                            pdf_file_path = found_path
                            break
        
        # 检查文件是否存在
        if not os.path.exists(pdf_file_path):
            # 列出目录中的所有PDF文件，帮助用户
            available_files = []
            if os.path.exists(DOWNLOAD_PAPERS_DIR):
                available_files = [f for f in os.listdir(DOWNLOAD_PAPERS_DIR) if f.endswith('.pdf')]
            
            return json.dumps({
                "success": False,
                "error": f"PDF文件不存在: {original_path}",
                "searched_path": pdf_file_path,
                "available_files": available_files[:5],  # 只返回前5个
                "note": f"请确保文件在统一目录 {DOWNLOAD_PAPERS_DIR} 中，或提供正确的文件名"
            }, indent=2, ensure_ascii=False)
        
        # 确定数据库类型（修复：处理空字符串）
        if not db_type or db_type.strip() == "":
            db_type = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()
        else:
            db_type = db_type.lower().strip()
        
        if db_type != "qdrant":
            return json.dumps({
                "success": False,
                "error": f"不支持的数据库类型: '{db_type}'。只支持: qdrant"
            }, indent=2, ensure_ascii=False)
        
        # 获取embeddings
        embeddings = get_embeddings()
        logger.info(f"使用 {db_type.upper()} 数据库创建向量数据库")
        
        # 确保统一文件夹存在
        os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
        
        # 生成数据库路径（修复：处理空字符串）
        if not db_name or db_name.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 从PDF文件名提取基础名称
            pdf_basename = os.path.splitext(os.path.basename(pdf_file_path))[0]
            # 清理文件名（移除特殊字符）
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', pdf_basename)[:50]
            db_name = f"{db_type}_{safe_name}_{timestamp}"
        else:
            db_name = db_name.strip()
        
        db_path = os.path.join(VECTOR_DB_BASE_DIR, db_name)
        
        # 如果数据库已存在，删除它
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            logger.info(f"已删除现有数据库: {db_path}")
        
        # 创建 Qdrant 向量存储
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "需要安装 qdrant-client 和 langchain-qdrant: pip install qdrant-client langchain-qdrant"
            }, indent=2, ensure_ascii=False)
        
        collection_name = "documents"
        client = QdrantClient(path=db_path)
        
        # 获取embedding维度
        embedding_dim = len(embeddings.embed_query("test"))
        
        # 创建集合
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        # 加载PDF文档
        logger.info(f"加载PDF文件: {pdf_file_path}")
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        
        if not documents:
            return json.dumps({
                "success": False,
                "error": "PDF文件为空或无法加载",
                "file_path": pdf_file_path
            }, indent=2, ensure_ascii=False)
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # 添加到向量数据库
        logger.info(f"正在添加 {len(chunks)} 个文本块到向量数据库...")
        vector_store.add_documents(chunks)
        
        # 更新环境变量，使新创建的数据库成为当前使用的数据库
        os.environ["VECTOR_DB_TYPE"] = "qdrant"
        os.environ["QDRANT_PATH"] = db_path
        os.environ["QDRANT_COLLECTION"] = collection_name
        
        # 清除缓存，强制重新加载
        clear_retriever_cache()
        
        return json.dumps({
            "success": True,
            "message": f"向量数据库创建成功",
            "db_path": db_path,
            "db_type": db_type,
            "pdf_file": pdf_file_path,
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "note": "数据库已自动切换为当前使用的数据库，Database_Search 工具现在可以使用它"
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"创建向量数据库时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "file_path": pdf_file_path
        }, indent=2, ensure_ascii=False)


create_vector_db_from_pdf: BaseTool = tool(create_vector_db_from_pdf_func)
create_vector_db_from_pdf.name = "Create_Vector_DB_From_PDF"


def get_vector_db_info_func() -> str:
    """
    获取当前向量数据库的信息。
    
    返回当前使用的向量数据库类型、路径、集合名（如果是Qdrant）等信息。
    也返回所有可用的向量数据库列表。
    
    Returns:
        str: JSON字符串，包含当前数据库信息和可用数据库列表
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 获取当前数据库配置
        db_type = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()
        
        default_path = os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant")
        db_path = os.getenv("QDRANT_PATH", default_path)
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        # 获取所有可用的数据库
        available_databases = []
        
        # 检查统一的向量数据库文件夹
        if os.path.exists(VECTOR_DB_BASE_DIR) and os.path.isdir(VECTOR_DB_BASE_DIR):
            for item in os.listdir(VECTOR_DB_BASE_DIR):
                db_path_item = os.path.join(VECTOR_DB_BASE_DIR, item)
                if os.path.isdir(db_path_item):
                    # 检测数据库类型
                    if os.path.exists(os.path.join(db_path_item, "config.json")):
                        available_databases.append({
                            "path": db_path_item,
                            "type": "qdrant",
                            "name": item
                        })
        
        # 兼容旧路径
        legacy_paths = [
            "./qdrant_db",
        ]
        
        for legacy_path in legacy_paths:
            if os.path.exists(legacy_path) and os.path.isdir(legacy_path):
                if os.path.exists(os.path.join(legacy_path, "config.json")):
                    db_type_legacy = "qdrant"
                else:
                    continue
                
                # 避免重复
                if not any(d["path"] == legacy_path for d in available_databases):
                    available_databases.append({
                        "path": legacy_path,
                        "type": db_type_legacy,
                        "name": os.path.basename(legacy_path)
                    })
        
        # 检查当前数据库是否存在
        current_db_exists = os.path.exists(db_path) if db_path else False
        
        result = {
            "current": {
                "db_type": db_type,
                "db_path": db_path,
                "collection_name": collection_name,
                "exists": current_db_exists
            },
            "available_databases": sorted(available_databases, key=lambda x: x["path"]),
            "total_available": len(available_databases)
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"获取向量数据库信息时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2, ensure_ascii=False)


get_vector_db_info: BaseTool = tool(get_vector_db_info_func)
get_vector_db_info.name = "Get_Vector_DB_Info"


def switch_vector_db_func(
    db_path: str,
    db_type: str = "",
    collection_name: str = ""
) -> str:
    """
    切换向量数据库。
    
    将当前使用的向量数据库切换到指定的数据库。切换后，Database_Search 工具将使用新的数据库。
    
    Args:
        db_path: 数据库路径（必须）
        db_type: 数据库类型（可选，如果不提供则自动检测，默认为 "qdrant"）
        collection_name: 集合名（可选，默认 "documents"）
    
    Returns:
        str: JSON字符串，包含切换结果信息
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 检查路径是否存在
        if not os.path.exists(db_path):
            return json.dumps({
                "success": False,
                "error": f"数据库路径不存在: {db_path}",
                "db_path": db_path
            }, indent=2, ensure_ascii=False)
        
        # 如果没有指定类型，尝试自动检测
        if not db_type or db_type.strip() == "":
            if os.path.exists(os.path.join(db_path, "config.json")):
                db_type = "qdrant"
            else:
                # 默认使用 Qdrant
                db_type = "qdrant"
        else:
            db_type = db_type.lower().strip()
        
        if db_type != "qdrant":
            return json.dumps({
                "success": False,
                "error": f"不支持的数据库类型: '{db_type}'。只支持: qdrant"
            }, indent=2, ensure_ascii=False)
        
        # 更新环境变量
        os.environ["VECTOR_DB_TYPE"] = "qdrant"
        os.environ["QDRANT_PATH"] = db_path
        os.environ["QDRANT_COLLECTION"] = collection_name or "documents"
        logger.info(f"切换到 Qdrant 数据库: {db_path} (集合: {collection_name or 'documents'})")
        
        # 清除缓存，强制重新加载
        clear_retriever_cache()
        
        return json.dumps({
            "success": True,
            "message": f"向量数据库已切换到: {db_path} (类型: {db_type})",
            "db_path": db_path,
            "db_type": db_type,
            "collection_name": collection_name or "documents",
            "note": "Database_Search 工具现在将使用新的数据库"
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"切换向量数据库时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "db_path": db_path
        }, indent=2, ensure_ascii=False)


switch_vector_db: BaseTool = tool(switch_vector_db_func)
switch_vector_db.name = "Switch_Vector_DB"

__all__ = [
    "database_search",
    "create_vector_db_from_pdf",
    "get_vector_db_info",
    "switch_vector_db",
]

