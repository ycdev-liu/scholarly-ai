"""文档处理相关的工具函数"""
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import UploadFile
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def save_uploaded_files(files: List[UploadFile]) -> List[Path]:
    """
    保存上传的文件到临时目录
    
    Args:
        files: 上传的文件列表
        
    Returns:
        保存的文件路径列表
    """
    saved_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for upload_file in files:
            file_path = Path(temp_dir) / upload_file.filename
            with open(file_path, "wb") as f:
                content = await upload_file.read()
                f.write(content)
            saved_files.append(file_path)
    return saved_files


def load_document(file_path: Path):
    """
    根据文件扩展名加载文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        Document loader 实例
        
    Raises:
        ValueError: 不支持的文件类型
    """
    filename = file_path.name
    
    if filename.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif filename.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif filename.endswith(".txt"):
        try:
            return TextLoader(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return TextLoader(file_path, encoding="gbk")
            except UnicodeDecodeError:
                return TextLoader(file_path, encoding="latin-1")
    else:
        raise ValueError(f"不支持的文件类型: {filename}")


def split_documents(documents, chunk_size: int = 2000, chunk_overlap: int = 500):
    """
    将文档分割成块
    
    Args:
        documents: 文档列表
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        文档块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def get_embeddings(use_local: bool = True, model_name: str = "BAAI/bge-m3"):
    """
    获取嵌入模型
    
    Args:
        use_local: 是否使用本地模型
        model_name: 模型名称
        
    Returns:
        嵌入模型实例
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    
    if use_local:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        cache_folder = os.path.join(os.getcwd(), "embedding.model")
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        # from langchain_openai import OpenAIEmbeddings
        # return OpenAIEmbeddings()
        raise NotImplementedError("OpenAI embeddings not implemented yet")
