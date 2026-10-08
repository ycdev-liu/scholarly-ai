"""共享工具函数和常量。"""
import os
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .hybrid_search import HybridRetriever

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
    contexts = []
    for doc in docs:
        metadata = doc.metadata
        source = Path(str(metadata.get("source", "unknown"))).name
        page_start = metadata.get("page_start", metadata.get("page"))
        page_end = metadata.get("page_end", page_start)
        if page_start is None:
            page_label = ""
        else:
            try:
                first = int(page_start) + 1
                last = int(page_end) + 1
                page_label = f", pages {first}-{last}" if last > first else f", page {first}"
            except (TypeError, ValueError):
                page_label = f", page {page_start}"
        section = metadata.get("section_path")
        section_line = f"\nSection: {section}" if section and not doc.page_content.startswith(f"{section}\n\n") else ""
        preview = metadata.get("preview_path")
        preview_line = f"\nPage preview: {preview}" if preview else ""
        contexts.append(f"Source: {source}{page_label}{section_line}\n{doc.page_content}{preview_line}")
    return "\n\n".join(contexts)


def get_embeddings():
    """获取当前配置的嵌入模型。"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        with _embeddings_lock:
            # 双重检查锁定
            if _embeddings_cache is None:
                provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
                if not provider:
                    provider = (
                        "local"
                        if os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"
                        else "openai"
                    )

                if provider == "dashscope":
                    from langchain_community.embeddings import DashScopeEmbeddings

                    api_key = os.getenv("DASHSCOPE_API_KEY")
                    if not api_key:
                        raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope embeddings")
                    _embeddings_cache = DashScopeEmbeddings(
                        model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
                        dashscope_api_key=api_key,
                    )
                elif provider == "local":
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
                elif provider == "openai":
                    try:
                        _embeddings_cache = OpenAIEmbeddings()
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
                        ) from e
                else:
                    raise ValueError(f"Unsupported embedding provider: {provider}")
    
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
    加载 Chroma 或 Qdrant，并统一包装为混合检索器。

    通过 VECTOR_DB_TYPE 选择数据库类型；未配置路径时使用默认知识库目录。
    """
    db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower() 
    embeddings = get_embeddings()
    
    # 确保统一文件夹存在
    os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)

    if db_type == "chroma":
        path = os.getenv("CHROMA_DB_PATH") or os.path.join(VECTOR_DB_BASE_DIR, "default_chroma")
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=path)
        # MMR 保留候选片段多样性；HybridRetriever 再与 BM25 结果做 RRF 融合。
        return HybridRetriever(
            vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 40}),
            vectorstore,
            "chroma",
            path,
        )
    if db_type != "qdrant":
        raise ValueError(f"Unsupported vector database type: {db_type}")
    
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
        
        return HybridRetriever(
            vector_store.as_retriever(search_kwargs={"k": 20}),
            vector_store,
            "qdrant",
            qdrant_path or os.path.join(VECTOR_DB_BASE_DIR, "default_qdrant"),
        )
