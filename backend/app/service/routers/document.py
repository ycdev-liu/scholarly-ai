"""文档处理相关的 API 路由"""
import gc
import os
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from agents.tools import clear_retriever_cache
from service.utils import verify_bearer
from service import document_processing

router = APIRouter(prefix="/api/documents", dependencies=[Depends(verify_bearer)])


def _delete_existing_db(db_path: Path, max_retries: int = 5) -> tuple[bool, str | None]:
    """
    删除已存在的数据库
    
    Args:
        db_path: 数据库路径
        max_retries: 最大重试次数
        
    Returns:
        (是否成功, 错误信息)
    """
    if not db_path.exists():
        return True, None
    
    try:
        # 清除可能正在使用的数据库连接缓存
        clear_retriever_cache()
        
        # 等待一小段时间，确保文件句柄被释放
        gc.collect()
        time.sleep(0.5)
        
        # 尝试删除，如果失败则重试（Windows 上文件锁定问题较常见）
        for attempt in range(max_retries):
            try:
                # 在 Windows 上，先尝试重命名再删除，有时可以绕过文件锁定
                if os.name == "nt":  # Windows
                    try:
                        # 尝试重命名为临时名称
                        temp_name = str(db_path) + f".deleting_{int(time.time())}"
                        if os.path.exists(temp_name):
                            shutil.rmtree(temp_name, ignore_errors=True)
                        os.rename(str(db_path), temp_name)
                        shutil.rmtree(temp_name, ignore_errors=True)
                        break
                    except (OSError, PermissionError):
                        # 如果重命名失败，直接尝试删除
                        shutil.rmtree(db_path)
                        break
                else:
                    shutil.rmtree(db_path)
                    break
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 0.5
                    time.sleep(wait_time)
                    gc.collect()
                else:
                    return False, f"无法删除旧数据库（文件可能被其他程序占用）: {str(e)}"
        return True, None
    except Exception as e:
        return False, f"无法删除旧数据库: {str(e)}"


@router.post("/upload", operation_id="upload_and_process_documents")
async def upload_and_process_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(2000),
    chunk_overlap: int = Form(500),
    use_local_embedding: bool = Form(True),
    model_name: str = Form("BAAI/bge-m3"),
    db_name: str = Form("chroma_db_uploader"),
    db_type: str = Form("qdrant"),
    auto_switch: bool = Form(False),
):
    """
    上传文件并处理文档，创建向量数据库
    
    Args:
        files: 上传的文件列表
        chunk_size: 文档块大小
        chunk_overlap: 文档块重叠大小
        use_local_embedding: 是否使用本地嵌入模型
        model_name: 嵌入模型名称
        db_name: 数据库名称
        db_type: 数据库类型 (qdrant, chroma, milvus)
        auto_switch: 是否自动切换到新创建的数据库
        
    Returns:
        处理结果
    """
    result = {
        "success": False,
        "db_name": db_name,
        "total_files": len(files),
        "total_chunks": 0,
        "processed_files": [],
        "errors": [],
    }
    
    try:
        # 1. 保存上传的文件
        saved_files = await document_processing.save_uploaded_files(files)
        
        # 2. 获取嵌入模型
        embeddings = document_processing.get_embeddings(use_local=use_local_embedding, model_name=model_name)
        
        # 3. 准备数据库路径
        VECTOR_DB_BASE_DIR = "./vector_databases"
        os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
        
        db_path = Path(db_name)
        if not db_path.is_absolute():
            if not str(db_path).startswith(VECTOR_DB_BASE_DIR):
                db_path = Path(VECTOR_DB_BASE_DIR) / db_path.name
            else:
                db_path = Path(os.getcwd()) / str(db_path)
        
        db_type = db_type.lower()
        
        # 4. 删除已存在的数据库
        success, error_msg = _delete_existing_db(db_path)
        if not success:
            result["errors"].append(error_msg)
            return result
        
        # 5. 创建向量存储（这部分委托给向量数据库模块）
        from service.routers import vectordb
        
        vector_store, collection_name = vectordb.create_vector_store(
            db_type=db_type,
            db_path=db_path,
            db_name=db_name,
            embeddings=embeddings,
        )
        
        if vector_store is None:
            result["errors"].append("无法创建向量存储，请检查数据库类型和依赖包")
            return result
        
        # 6. 处理文档并添加到向量存储
        for file_path in saved_files:
            filename = file_path.name
            try:
                # 加载文档
                loader = document_processing.load_document(file_path)
                documents = loader.load()
                
                # 分割文档
                chunks = document_processing.split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # 添加到向量存储
                if chunks:
                    vector_store.add_documents(chunks)
                    result["total_chunks"] += len(chunks)
                    result["processed_files"].append({"filename": filename, "chunks": len(chunks)})
            except ValueError as e:
                result["errors"].append(str(e))
                continue
            except Exception as e:
                result["errors"].append(f"处理文件 {filename} 时出错: {str(e)}")
                continue
        
        result["success"] = True
        result["db_path"] = str(db_path)
        result["db_type"] = db_type
        
        # 7. 自动切换数据库（如果需要）
        if auto_switch:
            try:
                from service.routers import vectordb
                
                switch_result = vectordb.switch_vector_db_internal(
                    db_type=db_type,
                    db_path=str(db_path),
                    collection_name=collection_name,
                )
                
                if switch_result["success"]:
                    result["switched"] = True
                    result["message"] = f"数据库创建成功并已自动切换到: {db_path}"
                else:
                    result["switched"] = False
                    result["switch_error"] = switch_result.get("error", "未知错误")
            except Exception as e:
                result["switched"] = False
                result["switch_error"] = str(e)
        else:
            result["auto_switch"] = False
            result["message"] = "向量数据库创建成功，但未自动切换"
    
    except Exception as e:
        result["errors"].append(f"处理文档时出错：{str(e)}")
    
    return result
