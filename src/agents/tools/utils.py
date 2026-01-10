"""
工具包共享工具函数
包含 embeddings、向量数据库加载等共享功能
"""
import os
import threading
import logging
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings

# 统一的数据存储基础目录
DATA_BASE_DIR = "./data"

# 统一的向量数据库文件夹（在基础目录下）
VECTOR_DB_BASE_DIR = os.path.join(DATA_BASE_DIR, "vector_databases")

# 统一的下载文件夹（在基础目录下）
DOWNLOAD_BASE_DIR = os.path.join(DATA_BASE_DIR, "downloads")
DOWNLOAD_PAPERS_DIR = os.path.join(DOWNLOAD_BASE_DIR, "papers")

# 全局缓存和锁
_embeddings_cache = None
_embeddings_lock = threading.Lock()
_vector_db_retriever = None
_vector_db_lock = threading.Lock()


def format_contexts(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_embeddings():
    """Get the embeddings for the local model or OpenAi model."""
    global _embeddings_cache
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
                    from core.settings import settings
                    # 使用统一的配置路径
                    cache_folder = settings.EMBEDDING_MODEL_CACHE_DIR
                    # 如果是相对路径，转换为绝对路径（相对于项目根目录）
                    if not os.path.isabs(cache_folder):
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                        cache_folder = os.path.join(project_root, cache_folder.lstrip("./"))
                    model_name = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-m3")  # 默认使用 bge-m3
                    
                    logger.info(f"使用本地模型: {model_name}")
                    logger.info(f"缓存文件夹: {cache_folder}")
                    
                    # 检查本地是否已有模型（优先使用本地模型）
                    model_dir_name = f"models--{model_name.replace('/', '--')}"
                    model_path = os.path.join(cache_folder, "hub", model_dir_name)
                    local_model_exists = os.path.exists(model_path) and os.path.isdir(model_path)
                    
                    if local_model_exists:
                        # 优先使用本地已下载的模型（离线模式）
                        logger.info(f"检测到本地已下载的模型: {model_path}")
                        logger.info("优先使用本地模型（离线模式）...")
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        try:
                            _embeddings_cache = HuggingFaceEmbeddings(
                                model_name=model_name,
                                cache_folder=cache_folder,
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True},
                            )
                            logger.info(f"✅ 成功从本地加载模型: {model_name}")
                        except Exception as offline_error:
                            logger.warning(f"本地模型加载失败: {offline_error}")
                            logger.info("尝试在线模式重新下载...")
                            # 如果本地加载失败，尝试在线模式
                            os.environ.pop("HF_HUB_OFFLINE", None)
                            try:
                                _embeddings_cache = HuggingFaceEmbeddings(
                                    model_name=model_name,
                                    cache_folder=cache_folder,
                                    model_kwargs={"device": "cpu"},
                                    encode_kwargs={"normalize_embeddings": True},
                                )
                                logger.info(f"✅ 在线模式成功加载模型: {model_name}")
                            except Exception as online_error:
                                logger.error(f"在线模式也失败: {online_error}")
                                # 回退到 OpenAI
                                if os.getenv("OPENAI_API_KEY"):
                                    logger.warning("本地模型加载失败，回退到 OpenAI Embeddings")
                                    os.environ.pop("HF_HUB_OFFLINE", None)
                                    _embeddings_cache = OpenAIEmbeddings()
                                    logger.info("已切换到 OpenAI Embeddings")
                                else:
                                    raise RuntimeError(
                                        f"无法加载模型 '{model_name}'：\n"
                                        f"- 本地模型路径: {model_path}\n"
                                        f"- 离线加载失败: {str(offline_error)[:200]}\n"
                                        f"- 在线下载失败: {str(online_error)[:200]}\n\n"
                                        f"解决方案：\n"
                                        f"1. 检查模型文件是否完整\n"
                                        f"2. 设置 OPENAI_API_KEY 使用 OpenAI Embeddings\n"
                                        f"3. 检查网络连接，重新下载模型"
                                    ) from online_error
                    else:
                        # 本地没有模型，尝试在线下载
                        logger.info(f"本地未找到模型，尝试从 HuggingFace 下载: {model_name}")
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        try:
                            _embeddings_cache = HuggingFaceEmbeddings(
                                model_name=model_name,
                                cache_folder=cache_folder,
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True},
                            )
                            logger.info(f"✅ 成功下载并加载模型: {model_name}")
                        except Exception as online_error:
                            logger.error(f"在线下载失败: {online_error}")
                            # 如果下载失败，检查是否有 OpenAI API key
                            if os.getenv("OPENAI_API_KEY"):
                                logger.warning("模型下载失败，回退到 OpenAI Embeddings")
                                _embeddings_cache = OpenAIEmbeddings()
                                logger.info("已切换到 OpenAI Embeddings")
                            else:
                                raise RuntimeError(
                                    f"无法下载模型 '{model_name}'：\n"
                                    f"- 错误: {str(online_error)[:200]}\n\n"
                                    f"解决方案：\n"
                                    f"1. 检查网络连接，确保可以访问 https://huggingface.co\n"
                                    f"2. 设置 OPENAI_API_KEY 使用 OpenAI Embeddings\n"
                                    f"3. 手动下载模型到: {cache_folder}"
                                ) from online_error
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
                # 对于 Qdrant，尝试关闭连接
                if hasattr(_vector_db_retriever, 'vectorstore'):
                    vectorstore = _vector_db_retriever.vectorstore
                    if hasattr(vectorstore, '_collection'):
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
    加载向量数据库（支持 Qdrant）
    通过环境变量 VECTOR_DB_TYPE 选择数据库类型
    默认路径统一使用 vector_databases 文件夹
    """
    db_type = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()  # 默认使用 Qdrant
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
        # 默认使用 Qdrant
        # 如果环境变量未设置，使用统一文件夹下的默认路径
        default_qdrant_path = os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant")
        qdrant_path = os.getenv("QDRANT_PATH", default_qdrant_path)
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "需要安装 qdrant-client 和 langchain-qdrant: "
                "pip install qdrant-client langchain-qdrant"
            )
        
        # 创建本地 Qdrant 客户端
        client = QdrantClient(path=qdrant_path)
        
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


# 导出 _get_retriever 供其他模块使用
__all__ = [
    "format_contexts",
    "get_embeddings",
    "load_vector_db",
    "clear_retriever_cache",
    "_get_retriever",
    "DATA_BASE_DIR",
    "VECTOR_DB_BASE_DIR",
    "DOWNLOAD_BASE_DIR",
    "DOWNLOAD_PAPERS_DIR",
]

