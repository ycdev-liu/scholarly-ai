"""共享工具函数和常量。"""
import os
import threading
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings

# 统一的数据存储基础目录
DATA_BASE_DIR = "./data"

# 统一的向量数据库文件夹（在基础目录下）
VECTOR_DB_BASE_DIR = os.path.join(DATA_BASE_DIR, "vector_databases")

# 统一的下载文件夹（在基础目录下）
DOWNLOAD_BASE_DIR = os.path.join(DATA_BASE_DIR, "downloads")
DOWNLOAD_PAPERS_DIR = os.path.join(DOWNLOAD_BASE_DIR, "papers")

# 全局变量
_vector_db_retriever = None
_vector_db_lock = threading.Lock()
_embeddings_cache = None
_embeddings_lock = threading.Lock()


def format_contexts(docs):
    """格式化检索到的文档。"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_embeddings():
    """获取本地模型或OpenAI模型的嵌入。"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        with _embeddings_lock:
            # 双重检查锁定
            if _embeddings_cache is None:
                use_local_model_env = os.getenv("USE_LOCAL_MODEL", "False")
                use_local_model = use_local_model_env.lower() == "true"
                
                if use_local_model:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    catche_folder = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "embedding.model",
                    )
                    model_name = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-small-en-v1.5")
                    
                    # 设置离线模式环境变量
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    
                    try:
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                    except Exception as e:
                        # 如果离线模式失败，尝试在线模式
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                else:
                    try:
                        _embeddings_cache = OpenAIEmbeddings()
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
                        ) from e
    
    return _embeddings_cache


def clear_retriever_cache():
    """
    清除向量数据库 retriever 缓存，并尝试关闭数据库连接
    """
    global _vector_db_retriever
    with _vector_db_lock:
        # 尝试关闭数据库连接（如果存在）
        if _vector_db_retriever is not None:
            try:
                # 对于 Chroma，尝试关闭连接
                if hasattr(_vector_db_retriever, 'vectorstore'):
                    vectorstore = _vector_db_retriever.vectorstore
                    if hasattr(vectorstore, '_client'):  # type: ignore[attr-defined]
                        # Chroma 客户端
                        if hasattr(vectorstore._client, 'close'):  # type: ignore[attr-defined]
                            vectorstore._client.close()  # type: ignore[attr-defined]
                    elif hasattr(vectorstore, '_collection'):  # type: ignore[attr-defined]
                        # Qdrant 客户端
                        if hasattr(vectorstore._collection, '_client'):  # type: ignore[attr-defined]
                            client = vectorstore._collection._client  # type: ignore[attr-defined]
                            if hasattr(client, 'close'):
                                client.close()
            except Exception as e:
                pass
        
        # 清除缓存
        _vector_db_retriever = None


def _get_retriever():
    """
    获取缓存的 retriever，如果不存在则创建。
    使用双重检查锁定模式确保线程安全。
    """
    global _vector_db_retriever
    
    if _vector_db_retriever is None:
        with _vector_db_lock:
            # 双重检查：再次检查是否已被其他线程创建
            if _vector_db_retriever is None:
                try:
                    _vector_db_retriever = load_vector_db()
                except Exception as e:
                    raise
    
    return _vector_db_retriever


def load_vector_db():
    """
    加载向量数据库和 Qdrant 
    通过环境变量 VECTOR_DB_TYPE 选择数据库类型
    默认路径统一使用 vector_databases 文件夹
    """
    db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower() 
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

        qdrant_url = os.getenv("QDRANT_URL")  # 例如: http://qdrant:6333
        qdrant_path = os.getenv("QDRANT_PATH")  # 本地路径
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")

        if qdrant_url:
            client = QdrantClient(url=qdrant_url)  # 远程模式
        else:
        # 本地嵌入式模式
            default_qdrant_path = os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant")
            qdrant_path = qdrant_path or default_qdrant_path
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
