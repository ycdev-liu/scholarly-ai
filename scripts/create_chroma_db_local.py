"""
下载本地embedding模型 n
使用本地开源 Embedding 模型创建 ChromaDB
并测试数据库的检索功能
支持多种本地 embedding 模型：
1. HuggingFaceEmbeddings (推荐)
2. OllamaEmbeddings (如果使用 Ollama)
3. SentenceTransformerEmbeddings

使用方法:
    # 使用 HuggingFace 模型
    python scripts/create_chroma_db_local.py --model huggingface --model-name BAAI/bge-small-en-v1.5

    # 使用 Ollama 模型
    python scripts/create_chroma_db_local.py --model ollama --model-name nomic-embed-text

    # 使用 SentenceTransformer 模型
    python scripts/create_chroma_db_local.py --model sentence-transformers --model-name all-MiniLM-L6-v2
"""
import argparse
import os
import shutil
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv()


def get_local_embeddings(model_type: str, model_name: str) -> Any:
    """
    获取本地 embedding 模型

    Args:
        model_type: 模型类型 ('huggingface', 'ollama', 'sentence-transformers')
        model_name: 模型名称

    Returns:
        Embeddings 实例
    """
    # 设置模型缓存目录为 embedding.model
    cache_folder = os.path.join(os.getcwd(), "embedding.model")
    
    if model_type == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            print(f"使用 HuggingFace 模型: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder,
                model_kwargs={"device": "cpu"},  # 使用 CPU，如果有 GPU 可以改为 "cuda"
                encode_kwargs={"normalize_embeddings": True},  # 归一化向量
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community: pip install langchain-community"
            )

    elif model_type == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings

            print(f"使用 Ollama 模型: {model_name}")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(
                model=model_name,
                base_url=ollama_base_url,
                cache_folder=cache_folder,
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community: pip install langchain-community"
            )

    elif model_type == "sentence-transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            # SentenceTransformer 实际上也是使用 HuggingFaceEmbeddings
            print(f"使用 SentenceTransformer 模型: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder, 
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community 和 sentence-transformers: "
                "pip install langchain-community sentence-transformers"
            )

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
    model_type: str = "huggingface",
    model_name: str = "BAAI/bge-small-en-v1.5",
):
    """
    使用本地 embedding 模型创建 ChromaDB

    Args:
        folder_path: 包含文档的文件夹路径
        db_name: 数据库名称/路径
        delete_chroma_db: 是否删除已存在的数据库
        chunk_size: 文本块大小
        overlap: 文本块重叠大小
        model_type: 模型类型 ('huggingface', 'ollama', 'sentence-transformers')
        model_name: 模型名称
    """
    # 获取本地 embedding 模型
    embeddings = get_local_embeddings(model_type, model_name)

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"已删除现有数据库: {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=db_name,
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    # 导入文档加载器
    try:
        from langchain_community.document_loaders import (
            Docx2txtLoader,
            PyPDFLoader,
            TextLoader,
        )
    except ImportError:
        raise ImportError("需要安装 langchain-community: pip install langchain-community")

    # Iterate over files in the folder
    total_chunks = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            # 尝试多种编码以支持不同格式的文本文件
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                # 如果 UTF-8 失败，尝试其他编码
                try:
                    loader = TextLoader(file_path, encoding="gbk")  # 中文 Windows 常用编码
                except UnicodeDecodeError:
                    loader = TextLoader(file_path, encoding="latin-1")  # 最后的后备选项
        else:
            print(f"跳过不支持的文件类型: {filename}")
            continue

        # Load and split document into chunks
        print(f"\n处理文件: {filename}")
        try:
            document = loader.load()
            chunks = text_splitter.split_documents(document)
            print(f"  分割为 {len(chunks)} 个文本块")

            # Add chunks to Chroma vector store
            chroma.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  ✅ 文档 {filename} 已添加到数据库")
        except Exception as e:
            print(f"  ❌ 处理文件 {filename} 时出错: {e}")
            continue

    print(f"\n✅ 向量数据库创建完成!")
    print(f"  位置: {db_name}")
    print(f"  总文本块数: {total_chunks}")
    print(f"  使用的模型: {model_name} ({model_type})")
    return chroma


def main():
    parser = argparse.ArgumentParser(description="使用本地 embedding 模型创建 ChromaDB")
    parser.add_argument(
        "--folder",
        type=str,
        default="./data",
        help="包含文档的文件夹路径 (默认: ./data)",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="./chroma_db",
        help="数据库名称/路径 (默认: ./chroma_db)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["huggingface", "ollama", "sentence-transformers"],
        default="huggingface",
        help="模型类型 (默认: huggingface)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="模型名称 (默认: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="文本块大小 (默认: 2000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=500,
        help="文本块重叠大小 (默认: 500)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="保留已存在的数据库（不删除）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("使用本地 Embedding 模型创建 ChromaDB")
    print("=" * 80)
    print(f"文档文件夹: {args.folder}")
    print(f"数据库路径: {args.db_name}")
    print(f"模型类型: {args.model}")
    print(f"模型名称: {args.model_name}")
    print("=" * 80)

    try:
        chroma = create_chroma_db(
            folder_path=args.folder,
            db_name=args.db_name,
            delete_chroma_db=not args.keep_existing,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_type=args.model,
            model_name=args.model_name,
        )

        # 测试检索
        print("\n" + "=" * 80)
        print("测试检索功能")
        print("=" * 80)
        retriever = chroma.as_retriever(search_kwargs={"k": 3})
        query = "What's the company's mission"
        print(f"查询: {query}")
        similar_docs = retriever.invoke(query)

        print(f"\n找到 {len(similar_docs)} 个相关文档:")
        for i, doc in enumerate(similar_docs, start=1):
            print(f"\n结果 {i}:")
            print(f"  内容: {doc.page_content[:200]}...")
            print(f"  来源: {doc.metadata.get('source', 'Unknown')}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

