"""向量数据库相关的 API 路由"""
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException

from agents.tools import clear_retriever_cache
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from typing import Any
from service.utils import verify_bearer

router = APIRouter(prefix="/api/vectordb", dependencies=[Depends(verify_bearer)])


def create_vector_store(
    db_type: str,
    db_path: Path,
    db_name: str,
    embeddings: Embeddings,
) -> tuple[Any, str | None]:
    """
    创建向量存储实例
    
    Args:
        db_type: 数据库类型 (qdrant)
        db_path: 数据库路径
        db_name: 数据库名称
        embeddings: 嵌入模型
    Returns:
        (向量存储实例, 集合名称) 或 (None, None) 如果创建失败
    """
    db_type = db_type.lower()
    collection_name = None
    
    
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        return None, None
    
    # 支持远程和本地模式
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url)  # 远程模式
    else:
        client = QdrantClient(path=str(db_path))  # 本地模式
    
    embedding_dim = len(embeddings.embed_query("test"))
    
    collection_name = "documents"
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store, collection_name
    



def switch_vector_db_internal(
    db_type: str,
    db_path: str,
    collection_name: Optional[str] = None,
) -> dict:
    """
    内部函数：切换向量数据库类型和路径
    
    Args:
        db_type: 数据库类型 ("chroma" 或 "qdrant")
        db_path: 数据库路径
        collection_name: 集合名（仅 Qdrant 需要，默认 "documents")
        
    Returns:
        切换结果字典
    """
    try:
        db_type = db_type.lower()
        if db_type not in ["chroma", "qdrant", "milvus"]:
            return {"success": False, "error": f"Invalid database type: {db_type}"}
        
        # 设置 VECTOR_DB_TYPE 环境变量
        os.environ["VECTOR_DB_TYPE"] = db_type
        
        if db_type == "qdrant":
            os.environ["QDRANT_PATH"] = db_path
            os.environ["QDRANT_COLLECTION"] = collection_name or "documents"
        elif db_type == "milvus":
            # Milvus 使用环境变量或连接参数
            pass
        else:  # chroma
            os.environ["CHROMA_DB_PATH"] = db_path
        
        # 清除缓存，强制重新加载
        clear_retriever_cache()
        
        return {
            "success": True,
            "message": f"Vector database switched to: {db_path} (type: {db_type})",
            "db_path": db_path,
            "db_type": db_type,
            "collection_name": collection_name if db_type == "qdrant" else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/switch", operation_id="switch_vector_db")
async def switch_vector_db(
    db_type: str = Form(...),
    db_path: str = Form(...),
    collection_name: Optional[str] = Form(None),
):
    """
    切换向量数据库类型和路径

    Args:
        db_type: 数据库类型 ("chroma" 或 "qdrant")
        db_path: 数据库路径
        collection_name: 集合名（仅 Qdrant 需要，默认 "documents")

    Returns:
        切换结果
    """
    try:
        db_type = db_type.lower()
        if db_type not in ["chroma", "qdrant", "milvus"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid database type. Supported types: chroma, qdrant, milvus",
            )
        
        result = switch_vector_db_internal(db_type, db_path, collection_name)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching vector database: {str(e)}")
