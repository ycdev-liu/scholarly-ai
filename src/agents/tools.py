import json
import math
import re
import time
import httpx
import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
import os

# 统一的数据存储基础目录
DATA_BASE_DIR = "./data"

# 统一的向量数据库文件夹（在基础目录下）
VECTOR_DB_BASE_DIR = os.path.join(DATA_BASE_DIR, "vector_databases")

# 统一的下载文件夹（在基础目录下）
DOWNLOAD_BASE_DIR = os.path.join(DATA_BASE_DIR, "downloads")
DOWNLOAD_PAPERS_DIR = os.path.join(DOWNLOAD_BASE_DIR, "papers")


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_embeddings():
    """Get the embeddings for the local model or OpenAi model."""
    global _embeddings_cache
    import logging
    logger = logging.getLogger(__name__)
    
    if _embeddings_cache is None:
        logger.debug("Embeddings cache is empty, creating new embeddings")
        with _embeddings_lock:
            # 双重检查锁定
            if _embeddings_cache is None:
                use_local_model_env = os.getenv("USE_LOCAL_MODEL", "False")
                use_local_model = use_local_model_env.lower() == "true"
                
                # 添加调试信息
                logger.info(f"USE_LOCAL_MODEL 环境变量值: {use_local_model_env}")
                logger.info(f"是否使用本地模型: {use_local_model}")
                
                if use_local_model:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    catche_folder = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "embedding.model",
                    )
                    model_name = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-small-en-v1.5")
                    
                    logger.info(f"使用本地模型: {model_name}")
                    logger.info(f"缓存文件夹: {catche_folder}")
                    
                    # 设置离线模式环境变量
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    
                    try:
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                        logger.info(f"Embeddings initialized successfully: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
                        # 如果离线模式失败，尝试在线模式
                        logger.warning("Retrying without offline mode...")
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                else:
                    logger.info("使用 OpenAI Embeddings")
                    try:
                        _embeddings_cache = OpenAIEmbeddings()
                        logger.info("OpenAI embeddings initialized successfully")
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
                        ) from e
    else:
        logger.debug("Using cached embeddings")
    
    return _embeddings_cache
import threading
_vector_db_retriever = None
_vector_db_lock = threading.Lock()
_embeddings_cache = None  # 添加这行
_embeddings_lock = threading.Lock()  # 添加这行


import logging
def clear_retriever_cache():
    """
    清除向量数据库 retriever 缓存，并尝试关闭数据库连接
    """
    logger = logging.getLogger(__name__)
    global _vector_db_retriever
    with _vector_db_lock:
        # 尝试关闭数据库连接（如果存在）
        if _vector_db_retriever is not None:
            try:
                # 对于 Chroma，尝试关闭连接
                if hasattr(_vector_db_retriever, 'vectorstore'):
                    vectorstore = _vector_db_retriever.vectorstore
                    if hasattr(vectorstore, '_client'):
                        # Chroma 客户端
                        if hasattr(vectorstore._client, 'close'):
                            vectorstore._client.close()
                    elif hasattr(vectorstore, '_collection'):
                        # Qdrant 客户端
                        if hasattr(vectorstore._collection, '_client'):
                            client = vectorstore._collection._client
                            if hasattr(client, 'close'):
                                client.close()
            except Exception as e:
                logger.warning(f"关闭数据库连接时出错（可忽略）: {e}")
        
        # 清除缓存
        _vector_db_retriever = None
        logger.info("Vector database retriever cache cleared")
    


def _get_retriever():
    """
    获取缓存的 retriever，如果不存在则创建。
    使用双重检查锁定模式确保线程安全。
    """
    global _vector_db_retriever
    
    logger = logging.getLogger(__name__)
    
    if _vector_db_retriever is None:
        logger.debug("Retriever cache is empty, creating new retriever")
        with _vector_db_lock:
            # 双重检查：再次检查是否已被其他线程创建
            if _vector_db_retriever is None:
                try:
                    logger.info("Loading vector database...")
                    _vector_db_retriever = load_vector_db()
                    logger.info("Vector database loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load vector database: {e}", exc_info=True)
                    raise
    else:
        logger.debug("Using cached retriever")
    
    return _vector_db_retriever


def load_vector_db():
    """
    加载向量数据库（支持 Chroma 和 Qdrant）
    通过环境变量 VECTOR_DB_TYPE 选择数据库类型
    默认路径统一使用 vector_databases 文件夹
    """
    db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()  # 默认使用 Chroma
    embeddings = get_embeddings()
    
    # 确保统一文件夹存在
    os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
    
    if db_type == "qdrant":
        # 使用 Qdrant 本地嵌入式模式
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "需要安装 qdrant-client 和 langchain-qdrant: "
                "pip install qdrant-client langchain-qdrant"
            )
        
        # Qdrant 本地嵌入式模式
        # 如果环境变量未设置，使用统一文件夹下的默认路径
        default_qdrant_path = os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant")
        qdrant_path = os.getenv("QDRANT_PATH", default_qdrant_path)
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        # 创建本地 Qdrant 客户端
        client = QdrantClient(path=qdrant_path)  # 本地嵌入式模式
        
        # 获取 embedding 维度
        embedding_dim = len(embeddings.embed_query("test"))
        
        # 确保集合存在
        try:
            client.get_collection(collection_name)
        except Exception:
            # 集合不存在，创建它
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        
        # 创建 Qdrant vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    
    else:
        # 默认使用 ChromaDB
        # 如果环境变量未设置，使用统一文件夹下的默认路径
        default_chroma_path = os.path.join(VECTOR_DB_BASE_DIR, "default_chroma")
        db_path = os.getenv("CHROMA_DB_PATH", default_chroma_path)
        chroma_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
        return retriever





def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    import logging
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
database_search.name = "Database_Search"  # Update name with the purpose of your database



def create_vector_db_from_pdf_func(
    pdf_file_path: str,
    db_name: str = "",
    db_type: str = "",
    chunk_size: int = 2000,
    chunk_overlap: int = 500,
) -> str:
    """
    从PDF文件创建向量数据库。
    
    这个工具可以从下载的论文PDF文件创建向量数据库，支持ChromaDB和Qdrant。
    创建后，数据库可以被 Database_Search 工具使用。
    数据库默认保存在统一的数据目录下的 vector_databases 文件夹。
    
    Args:
        pdf_file_path: PDF文件的完整路径或文件名（例如：./data/downloads/papers/paper_xxx.pdf）
                      如果只提供文件名，会在统一目录 ./data/downloads/papers/ 中查找
                      如果文件名不存在，会尝试在目录中查找包含关键词的PDF文件
        db_name: 数据库名称（可选，如果不提供或为空则自动生成，基于时间戳）
        db_type: 数据库类型 "chroma" 或 "qdrant"（可选，默认使用环境变量 VECTOR_DB_TYPE）
        chunk_size: 文本块大小，默认 2000
        chunk_overlap: 文本块重叠大小，默认 500
    
    Returns:
        str: JSON字符串，包含创建结果信息
    """
    import logging
    import shutil
    from datetime import datetime
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    
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
            db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
        else:
            db_type = db_type.lower().strip()
        
        if db_type not in ["chroma", "qdrant"]:
            return json.dumps({
                "success": False,
                "error": f"不支持的数据库类型: '{db_type}'。支持的类型: chroma, qdrant"
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
        
        # 根据数据库类型创建向量存储
        if db_type == "qdrant":
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
        else:
            # ChromaDB
            from langchain_chroma import Chroma
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=db_path
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
        os.environ["VECTOR_DB_TYPE"] = db_type
        if db_type == "qdrant":
            os.environ["QDRANT_PATH"] = db_path
            os.environ["QDRANT_COLLECTION"] = collection_name
        else:
            os.environ["CHROMA_DB_PATH"] = db_path
        
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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 获取当前数据库配置
        db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
        
        if db_type == "qdrant":
            default_path = os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant")
            db_path = os.getenv("QDRANT_PATH", default_path)
            collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        else:
            default_path = os.path.join(VECTOR_DB_BASE_DIR, "default_chroma")
            db_path = os.getenv("CHROMA_DB_PATH", default_path)
            collection_name = None
        
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
                    elif any(
                        os.path.exists(os.path.join(db_path_item, check_item))
                        for check_item in ["chroma.sqlite3", "chroma.sqlite3-wal", "index"]
                    ):
                        available_databases.append({
                            "path": db_path_item,
                            "type": "chroma",
                            "name": item
                        })
        
        # 兼容旧路径
        legacy_paths = [
            "./chroma_db",
            "./chroma_db_mixed",
            "./chroma_db_uploaded",
            "./qdrant_db",
        ]
        
        for legacy_path in legacy_paths:
            if os.path.exists(legacy_path) and os.path.isdir(legacy_path):
                if os.path.exists(os.path.join(legacy_path, "config.json")):
                    db_type_legacy = "qdrant"
                elif any(
                    os.path.exists(os.path.join(legacy_path, check_item))
                    for check_item in ["chroma.sqlite3", "chroma.sqlite3-wal", "index"]
                ):
                    db_type_legacy = "chroma"
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
        db_type: 数据库类型 "chroma" 或 "qdrant"（可选，如果不提供则自动检测）
        collection_name: 集合名（仅Qdrant需要，可选，默认 "documents"）
    
    Returns:
        str: JSON字符串，包含切换结果信息
    """
    import logging
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
            elif any(
                os.path.exists(os.path.join(db_path, check_item))
                for check_item in ["chroma.sqlite3", "chroma.sqlite3-wal", "index"]
            ):
                db_type = "chroma"
            else:
                return json.dumps({
                    "success": False,
                    "error": f"无法自动检测数据库类型，请指定 db_type 参数",
                    "db_path": db_path
                }, indent=2, ensure_ascii=False)
        else:
            db_type = db_type.lower().strip()
        
        if db_type not in ["chroma", "qdrant"]:
            return json.dumps({
                "success": False,
                "error": f"不支持的数据库类型: '{db_type}'。支持的类型: chroma, qdrant"
            }, indent=2, ensure_ascii=False)
        
        # 更新环境变量
        os.environ["VECTOR_DB_TYPE"] = db_type
        
        if db_type == "qdrant":
            os.environ["QDRANT_PATH"] = db_path
            os.environ["QDRANT_COLLECTION"] = collection_name or "documents"
            logger.info(f"切换到 Qdrant 数据库: {db_path} (集合: {collection_name or 'documents'})")
        else:
            os.environ["CHROMA_DB_PATH"] = db_path
            logger.info(f"切换到 ChromaDB 数据库: {db_path}")
        
        # 清除缓存，强制重新加载
        clear_retriever_cache()
        
        return json.dumps({
            "success": True,
            "message": f"向量数据库已切换到: {db_path} (类型: {db_type})",
            "db_path": db_path,
            "db_type": db_type,
            "collection_name": collection_name if db_type == "qdrant" else None,
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


def openreview_search_func(
    venue: str = "",
    domain: str = "",
    invitation: str = "",
    keyword: str = "",
    max_papers: int = 100,
    limit_per_page: int = 25,
) -> str:
    """Searches OpenReview API for academic papers.
    
    Useful for when you need to find papers from conferences like ICML, NeurIPS, ICLR, etc.
    This tool fetches papers from OpenReview API with pagination support.
    You can search by venue or by keyword. If keyword is provided, it will search across 
    multiple conferences and filter results by title/abstract containing the keyword.
    
    Args:
        venue (str): The venue filter (e.g., "ICML 2025 oral", "NeurIPS 2024"). 
                     If empty, will search across multiple conferences.
        domain (str): The conference domain (e.g., "ICML.cc/2025/Conference"). 
                     If empty and venue is provided, will try to infer from venue.
        invitation (str): The submission invitation path. If empty, will use default.
        keyword (str): Keyword to search for in paper titles/abstracts. 
                       If provided, will filter results to papers containing this keyword.
        max_papers (int): Maximum number of papers to fetch. Default: 100
        limit_per_page (int): Number of papers per page. Default: 25
    
    Returns:
        str: JSON string containing the list of papers with their details.
    """
    base_url = "https://api2.openreview.net/notes"
    all_notes = []
    
    # 如果没有指定 venue，尝试搜索多个常见会议
    venues_to_try = []
    if venue:
        venues_to_try = [(venue, domain or "", invitation or "")]
    else:
        # 默认搜索多个会议
        venues_to_try = [
            ("NeurIPS 2024", "NeurIPS.cc/2024/Conference", "NeurIPS.cc/2024/Conference/-/Submission"),
            ("ICLR 2024", "ICLR.cc/2024/Conference", "ICLR.cc/2024/Conference/-/Submission"),
            ("ICML 2024", "ICML.cc/2024/Conference", "ICML.cc/2024/Conference/-/Submission"),
            ("NeurIPS 2023", "NeurIPS.cc/2023/Conference", "NeurIPS.cc/2023/Conference/-/Submission"),
        ]
    
    try:
        for venue_name, venue_domain, venue_invitation in venues_to_try:
            if not venue_domain:
                continue
                
            for offset in range(0, max_papers, limit_per_page):
                params = {
                    "content.venue": venue_name,
                    "details": "replyCount,presentation,writable",
                    "domain": venue_domain,
                    "invitation": venue_invitation,
                    "limit": limit_per_page,
                    "offset": offset,
                }
                
                try:
                    response = httpx.get(base_url, params=params, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get("notes"):
                        break
                    
                    # 如果有关键词，过滤结果
                    if keyword:
                        filtered_notes = []
                        keyword_lower = keyword.lower()
                        for note in data["notes"]:
                            title = str(note.get("content", {}).get("title", "")).lower()
                            abstract = str(note.get("content", {}).get("abstract", "")).lower()
                            if keyword_lower in title or keyword_lower in abstract:
                                filtered_notes.append(note)
                        all_notes.extend(filtered_notes)
                    else:
                        all_notes.extend(data["notes"])
                    
                    # 如果已经找到足够的论文，停止搜索
                    if len(all_notes) >= max_papers:
                        break
                    
                    # Rate limiting
                    time.sleep(0.5)
                except httpx.HTTPError as e:
                    # 如果某个会议搜索失败，继续尝试下一个
                    logger.warning(f"Failed to search {venue_name}: {e}")
                    break
            
            # 如果已经找到足够的论文，停止搜索其他会议
            if len(all_notes) >= max_papers:
                break
        
        # 限制返回的论文数量
        all_notes = all_notes[:max_papers]
        
        # Format the results
        result = {
            "total_papers": len(all_notes),
            "papers": [
                {
                    "id": note.get("id"),
                    "title": note.get("content", {}).get("title", "N/A"),
                    "abstract": note.get("content", {}).get("abstract", "N/A"),
                    "authors": note.get("content", {}).get("authors", []),
                    "venue": note.get("content", {}).get("venue", "N/A"),
                    "keywords": note.get("content", {}).get("keywords", []),
                    "pdf_url": f"https://openreview.net/pdf?id={note.get('id')}" if note.get("id") else None,
                    "openreview_url": f"https://openreview.net/forum?id={note.get('id')}" if note.get("id") else None,
                }
                for note in all_notes
            ],
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error processing OpenReview data: {e}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "total_papers": 0,
            "papers": [],
            "error": error_msg
        }, indent=2, ensure_ascii=False)


openreview_search: BaseTool = tool(openreview_search_func)
openreview_search.name = "OpenReview_Search"


def download_paper_func(
    paper_id: str,
    save_path: str = "",
    download_dir: str = ""
) -> str:
    """下载 OpenReview 论文的 PDF 文件。
    
    从 OpenReview 下载指定论文的 PDF 文件并保存到本地。
    默认保存到统一的数据目录下的 downloads/papers 文件夹。
    
    Args:
        paper_id: OpenReview 论文 ID（例如从 openreview_search 获取的 id）
        save_path: 保存文件的完整路径（可选，如果不提供则自动生成）
        download_dir: 下载目录（可选，如果不提供则使用统一目录 ./data/downloads/papers）
    
    Returns:
        str: JSON 字符串，包含下载结果信息
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 确保下载目录存在（使用统一路径）
        if download_dir == "":
            # 优先使用环境变量，如果没有则使用统一目录
            # 注意：如果环境变量指向旧路径，也使用统一目录
            env_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "")
            if env_dir and env_dir != "./downloads" and env_dir != "./downloads/papers":
                # 如果环境变量设置了且不是旧路径，使用环境变量
                download_dir = env_dir
            else:
                # 否则使用统一目录
                download_dir = DOWNLOAD_PAPERS_DIR
        os.makedirs(download_dir, exist_ok=True)
        
        # 构建 PDF URL
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"
        
        # 如果没有指定保存路径，自动生成
        if not save_path:
            # 先获取论文标题作为文件名
            try:
                # 获取论文详细信息以获取标题
                detail_url = f"https://api2.openreview.net/notes?id={paper_id}"
                detail_response = httpx.get(detail_url, timeout=30.0)
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                
                if detail_data.get("notes"):
                    title = detail_data["notes"][0].get("content", {}).get("title", "paper")
                    # 清理文件名（移除非法字符）
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]  # 限制长度
                    filename = f"{safe_title}_{paper_id[:8]}.pdf"
                else:
                    filename = f"paper_{paper_id[:8]}.pdf"
            except Exception:
                # 如果获取标题失败，使用默认文件名
                filename = f"paper_{paper_id[:8]}.pdf"
            
            save_path = os.path.join(download_dir, filename)
        
        # 下载 PDF
        logger.info(f"正在下载论文 PDF: {pdf_url}")
        response = httpx.get(pdf_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        
        # 检查是否是 PDF 内容
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
            # 可能返回的是 HTML 页面（论文未公开），尝试其他方式
            raise ValueError(f"无法下载 PDF，可能论文未公开或需要登录。Content-Type: {content_type}")
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024  # KB
        
        return json.dumps({
            "success": True,
            "paper_id": paper_id,
            "file_path": save_path,
            "file_size_kb": round(file_size, 2),
            "message": f"论文 PDF 已成功下载到: {save_path}"
        }, indent=2, ensure_ascii=False)
        
    except httpx.HTTPError as e:
        error_msg = f"下载论文时 HTTP 错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "paper_id": paper_id,
            "error": error_msg
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        error_msg = f"下载论文时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "paper_id": paper_id,
            "error": error_msg
        }, indent=2, ensure_ascii=False)


download_paper: BaseTool = tool(download_paper_func)
download_paper.name = "Download_Paper"


def download_paper_from_arxiv_func(
    arxiv_id: str = "",
    arxiv_url: str = "",
    save_path: str = "",
    download_dir: str = ""
) -> str:
    """从 arXiv 下载论文的 PDF 文件。
    
    支持通过 arXiv ID 或 URL 下载论文。
    例如：arxiv_id="1706.03762" 或 arxiv_url="https://arxiv.org/abs/1706.03762"
    
    Args:
        arxiv_id: arXiv 论文 ID（例如：1706.03762，不需要 "arXiv:" 前缀）
        arxiv_url: arXiv 论文 URL（例如：https://arxiv.org/abs/1706.03762）
        save_path: 保存文件的完整路径（可选，如果不提供则自动生成）
        download_dir: 下载目录（可选，如果不提供则使用统一目录 ./data/downloads/papers）
    
    Returns:
        str: JSON 字符串，包含下载结果信息
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 从 URL 或 ID 提取 arXiv ID
        if arxiv_url:
            # 从 URL 提取 ID
            import re
            match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', arxiv_url)
            if match:
                arxiv_id = match.group(1)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"无法从 URL 中提取 arXiv ID: {arxiv_url}"
                }, indent=2, ensure_ascii=False)
        
        if not arxiv_id:
            return json.dumps({
                "success": False,
                "error": "必须提供 arxiv_id 或 arxiv_url"
            }, indent=2, ensure_ascii=False)
        
        # 清理 arXiv ID（移除 "arXiv:" 前缀等）
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "").strip()
        
        # 确保下载目录存在
        if download_dir == "":
            env_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "")
            if env_dir and env_dir != "./downloads" and env_dir != "./downloads/papers":
                download_dir = env_dir
            else:
                download_dir = DOWNLOAD_PAPERS_DIR
        os.makedirs(download_dir, exist_ok=True)
        
        # 构建 arXiv PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # 如果没有指定保存路径，自动生成
        if not save_path:
            # 尝试获取论文标题
            try:
                abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                abs_response = httpx.get(abs_url, timeout=30.0)
                abs_response.raise_for_status()
                
                # 从 HTML 中提取标题
                import re
                title_match = re.search(r'<title>([^<]+)</title>', abs_response.text)
                if title_match:
                    title = title_match.group(1).replace("arXiv:", "").strip()
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]
                    filename = f"{safe_title}_{arxiv_id}.pdf"
                else:
                    filename = f"arxiv_{arxiv_id}.pdf"
            except Exception:
                filename = f"arxiv_{arxiv_id}.pdf"
            
            save_path = os.path.join(download_dir, filename)
        
        # 下载 PDF
        logger.info(f"正在从 arXiv 下载论文 PDF: {pdf_url}")
        response = httpx.get(pdf_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        
        # 检查是否是 PDF 内容
        if not response.content.startswith(b"%PDF"):
            return json.dumps({
                "success": False,
                "error": f"下载的内容不是 PDF 文件，可能 arXiv ID 不正确: {arxiv_id}"
            }, indent=2, ensure_ascii=False)
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024  # KB
        
        return json.dumps({
            "success": True,
            "arxiv_id": arxiv_id,
            "file_path": save_path,
            "file_size_kb": round(file_size, 2),
            "message": f"论文 PDF 已成功下载到: {save_path}"
        }, indent=2, ensure_ascii=False)
        
    except httpx.HTTPError as e:
        error_msg = f"从 arXiv 下载论文时 HTTP 错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "arxiv_id": arxiv_id if 'arxiv_id' in locals() else "",
            "error": error_msg
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        error_msg = f"从 arXiv 下载论文时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "arxiv_id": arxiv_id if 'arxiv_id' in locals() else "",
            "error": error_msg
        }, indent=2, ensure_ascii=False)


download_paper_from_arxiv: BaseTool = tool(download_paper_from_arxiv_func)
download_paper_from_arxiv.name = "Download_Paper_From_ArXiv"


def list_downloaded_papers_func() -> str:
    """
    列出已下载的论文文件。
    
    返回 ./data/downloads/papers/ 目录中的所有PDF文件列表。
    
    Returns:
        str: JSON字符串，包含文件列表
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        papers = []
        if os.path.exists(DOWNLOAD_PAPERS_DIR):
            for file in os.listdir(DOWNLOAD_PAPERS_DIR):
                if file.endswith('.pdf'):
                    file_path = os.path.join(DOWNLOAD_PAPERS_DIR, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    papers.append({
                        "filename": file,
                        "path": file_path,
                        "size_kb": round(file_size, 2)
                    })
        
        return json.dumps({
            "success": True,
            "directory": DOWNLOAD_PAPERS_DIR,
            "total_files": len(papers),
            "papers": sorted(papers, key=lambda x: x["filename"])
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"列出文件时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2, ensure_ascii=False)


list_downloaded_papers: BaseTool = tool(list_downloaded_papers_func)
list_downloaded_papers.name = "List_Downloaded_Papers"


def test_download_and_create_vector_db():
    """
    测试：使用工具函数完成搜索、下载和创建向量数据库的完整流程
    
    流程：
    1. 使用 openreview_search_func 搜索论文
    2. 使用 download_paper_func 下载论文PDF
    3. 从下载的PDF创建向量数据库
    """
    import json
    import shutil
    from datetime import datetime
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from dotenv import load_dotenv
    
    # 加载 .env 文件
    load_dotenv()
    
    print("=" * 80)
    print("测试：使用工具函数完成搜索、下载和创建向量数据库")
    print("=" * 80)
    
    # 步骤1: 使用 openreview_search_func 搜索论文
    print("\n[步骤 1/3] 使用 openreview_search_func 搜索论文...")
    try:
        # 使用工具函数搜索论文
        search_result = openreview_search_func(max_papers=1, limit_per_page=1)
        search_data = json.loads(search_result)
        
        if not search_data.get("papers") or len(search_data["papers"]) == 0:
            print("❌ 未找到论文，无法继续测试")
            return
        
        paper = search_data["papers"][0]
        paper_id = paper.get("id")
        
        # 确保 paper_title 是字符串（处理可能的类型问题）
        paper_title_raw = paper.get("title", "Unknown")
        if isinstance(paper_title_raw, dict):
            # 如果是字典，尝试提取文本
            paper_title = paper_title_raw.get("text", 
                          paper_title_raw.get("value", 
                          str(paper_title_raw)))
        elif isinstance(paper_title_raw, str):
            paper_title = paper_title_raw
        else:
            paper_title = str(paper_title_raw) if paper_title_raw else "Unknown"
        
        print(f"✅ 找到论文: {paper_title}")
        print(f"   Paper ID: {paper_id}")
        print(f"   Venue: {paper.get('venue', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 搜索论文失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: 使用 download_paper_func 下载论文
    print("\n[步骤 2/3] 使用 download_paper_func 下载论文PDF...")
    try:
        # 使用统一目录，或环境变量指定的下载目录
        download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", DOWNLOAD_PAPERS_DIR)
        
        # 使用工具函数下载论文
        download_result = download_paper_func(
            paper_id=paper_id, 
            download_dir=download_dir
        )
        download_data = json.loads(download_result)
        
        if not download_data.get("success"):
            print(f"❌ 下载失败: {download_data.get('error', 'Unknown error')}")
            return
        
        file_path = download_data.get("file_path")
        file_size = download_data.get("file_size_kb", 0)
        
        print(f"✅ 下载成功!")
        print(f"   文件路径: {file_path}")
        print(f"   文件大小: {file_size} KB")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return
            
    except Exception as e:
        print(f"❌ 下载论文失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 创建向量数据库
    print("\n[步骤 3/3] 从PDF创建向量数据库...")
    try:
        # 获取embeddings（使用 get_embeddings 工具函数）
        embeddings = get_embeddings()
        print(f"✅ Embeddings 模型已加载")
        
        # 确定数据库类型和路径
        db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if db_type == "qdrant":
            # 使用 Qdrant
            try:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError:
                print("❌ 需要安装 qdrant-client 和 langchain-qdrant")
                return
            
            db_path = os.path.join(VECTOR_DB_BASE_DIR, f"qdrant_papers_{timestamp}")
            collection_name = "papers"
            
            # 如果数据库已存在，删除它
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            # 创建 Qdrant 客户端
            client = QdrantClient(path=db_path)
            
            # 获取 embedding 维度
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
            
            # 创建 Qdrant vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            
            print(f"✅ Qdrant 数据库已创建: {db_path}")
            
        else:
            # 使用 ChromaDB
            db_path = os.path.join(VECTOR_DB_BASE_DIR, f"chroma_papers_{timestamp}")
            
            # 如果数据库已存在，删除它
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            from langchain_chroma import Chroma
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=db_path
            )
            
            print(f"✅ ChromaDB 数据库已创建: {db_path}")
        
        # 加载PDF文档
        print(f"\n   加载PDF文件: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"   ✅ 加载了 {len(documents)} 页")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   ✅ 分割为 {len(chunks)} 个文本块")
        
        # 添加到向量数据库（分批处理，显示进度）
        print(f"\n   正在添加到向量数据库...")
        print(f"   提示: 这可能需要一些时间，请耐心等待...")
        
        import time
        batch_size = 10  # 每批处理10个chunks
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            progress = (batch_num / total_batches) * 100
            elapsed_time = time.time() - start_time
            
            if batch_num > 1:
                avg_time = elapsed_time / batch_num
                remaining = avg_time * (total_batches - batch_num)
                print(f"   进度: {batch_num}/{total_batches} ({progress:.1f}%) | "
                      f"已用时: {elapsed_time:.0f}s | 预计剩余: {remaining:.0f}s", end='\r')
            else:
                print(f"   进度: {batch_num}/{total_batches} ({progress:.1f}%)", end='\r')
            
            vector_store.add_documents(batch)
        
        total_time = time.time() - start_time
        print(f"\n   ✅ 已添加到向量数据库 (用时: {total_time:.1f}秒)")
        
        # 测试检索（使用 database_search_func 的逻辑）
        print(f"\n   测试检索功能...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 确保 test_query 是字符串
        if isinstance(paper_title, str) and paper_title:
            # 取标题的前几个词作为测试查询
            words = paper_title.split()
            test_query = " ".join(words[:3]) if len(words) >= 3 else words[0] if words else "machine learning"
        else:
            test_query = "machine learning"
        
        results = retriever.invoke(test_query)
        print(f"   ✅ 检索测试成功，找到 {len(results)} 个相关文档块")
        
        # 显示检索结果摘要
        if results:
            print(f"\n   检索结果预览（查询: '{test_query}'）:")
            for i, doc in enumerate(results[:2], 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i}. {preview}...")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        print(f"论文标题: {paper_title}")
        print(f"PDF文件: {file_path}")
        print(f"向量数据库: {db_path} (类型: {db_type.upper()})")
        print(f"文本块数: {len(chunks)}")
        print(f"检索测试查询: '{test_query}'")
        print(f"检索结果数: {len(results)}")
        
    except Exception as e:
        print(f"❌ 创建向量数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    # test_tools.py - 放在项目根目录
    import json
    import sys
    from pathlib import Path
    from dotenv import load_dotenv  # 添加这行
    
    # 加载 .env 文件
    load_dotenv()
    
    # 调试：显示环境变量值
    use_local_model = os.getenv("USE_LOCAL_MODEL", "False")
    print(f"[调试信息] USE_LOCAL_MODEL = {use_local_model}")
    print(f"[调试信息] 将使用: {'本地模型' if use_local_model.lower() == 'true' else 'OpenAI模型'}")

    project_root = Path(__file__).parent / "agent-service-toolkit"
    sys.path.insert(0, str(project_root / "src"))

    # 测试搜索
    print("测试搜索功能...")
    result = openreview_search_func(max_papers=3, limit_per_page=3)
    data = json.loads(result)
    print(f"找到 {data['total_papers']} 篇论文")
    if data['papers']:
        paper = data['papers'][0]
        print(f"第一篇论文: {paper['title']}")
        print(f"PDF URL: {paper.get('pdf_url', 'N/A')}")
        
        # 测试下载
        if paper.get('id'):
            print(f"\n测试下载功能...")
            download_result = download_paper_func(paper_id=paper['id'], download_dir="")
            print(download_result)
    
    # 新增：测试下载并创建向量数据库
    print("\n" + "=" * 80)
    print("开始测试：下载论文并创建向量数据库")
    print("=" * 80)
    test_download_and_create_vector_db()

if __name__ == "__main__":
    main()