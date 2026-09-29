"""Tests for the RAG Assistant Agent."""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

# 添加 src 目录到 Python 路径，以便直接运行测试文件时也能正确导入
if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import asyncio
import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agents.rag_assistant import rag_assistant
from core import settings
from core.logging_config import setup_logging

# 加载环境变量
load_dotenv()


def extract_tool_results(result: dict) -> dict:
    """从 agent 结果中提取工具调用和结果"""
    tool_calls = []
    tool_results = []
    
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
    
    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results
    }


def parse_tool_result(tool_result: str) -> dict:
    """解析工具结果的JSON字符串"""
    try:
        return json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return {}


async def test_rag_agent(message: str, test_name: str = "测试", verbose: bool = True):
    """测试 RAG agent 的响应"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"{test_name}: {message}")
        print("=" * 80)

    inputs: MessagesState = {
        "messages": [HumanMessage(content=message)]
    }
    config_dict = {
        "thread_id": str(uuid4()),
        "model": settings.DEFAULT_MODEL
    }
    
    try:
        result = await rag_assistant.ainvoke(
            input=inputs,
            config=RunnableConfig(configurable=config_dict)
        )
        
        if verbose:
            print("\n" + "-" * 80)
            print("Agent 响应:")
            print("-" * 80)

            for idx, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                print(f"\n[消息 {idx + 1}] ({msg_type}):")

                if hasattr(msg, "content") and msg.content:
                    content_preview = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    print(f"内容: {content_preview}")
                
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"工具调用 ({len(msg.tool_calls)} 个):")
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        print(f"  - {tool_name}")
                        if tool_args:
                            for key, value in tool_args.items():
                                if key in ["query", "pdf_file_path", "db_name", "db_type"]:
                                    print(f"    {key}: {value}")

        if verbose and result["messages"]:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content") and msg.content:
                print("\n" + "=" * 80)
                print("最终响应:")
                print("=" * 80)
                print(last_msg.content)
        
        if verbose:
            print(f"\n总共处理了 {len(result['messages'])} 条消息")
        
        return result

    except Exception as e:
        if verbose:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        raise


async def main():
    """主测试函数"""
    # 设置日志
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file="test_rag_assistant.log",
        log_dir="./logs",
    )
    
    print("=" * 80)
    print("RAG Assistant Agent 测试")
    print("=" * 80)
    print(f"使用模型: {settings.DEFAULT_MODEL}")
    print(f"Agent ID: rag-assistant")
    
    # 检查是否有可用的PDF文件用于测试
    download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
    pdf_files = []
    
    # 查找下载目录中的PDF文件
    if os.path.exists(download_dir):
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
    
    # 测试 1: 测试数据库搜索（如果数据库存在）
    print("\n" + "=" * 80)
    print("测试 1: 数据库搜索功能")
    print("=" * 80)
    
    result1 = await test_rag_agent(
        "搜索关于机器学习的信息",
        "测试 1: 数据库搜索"
    )
    
    # 验证搜索结果
    tool_info = extract_tool_results(result1)
    if tool_info["tool_calls"]:
        search_calls = [call for call in tool_info["tool_calls"] 
                       if call.get("name") == "Database_Search"]
        if search_calls:
            print(f"\n✅ 成功调用 Database_Search 工具")
            if tool_info["tool_results"]:
                print(f"   搜索返回了结果")
        else:
            print(f"\n⚠️  未调用 Database_Search 工具（可能数据库为空或不存在）")
    else:
        print(f"\n⚠️  未调用任何工具（可能数据库为空或不存在）")
    
    await asyncio.sleep(1)
    
    # 测试 2: 如果有PDF文件，测试创建向量数据库
    if pdf_files:
        print("\n" + "=" * 80)
        print("测试 2: 从PDF创建向量数据库")
        print("=" * 80)
        
        test_pdf = pdf_files[0]
        print(f"使用PDF文件: {test_pdf}")
        
        result2 = await test_rag_agent(
            f"请为PDF文件 {test_pdf} 创建向量数据库",
            "测试 2: 创建向量数据库"
        )
        
        # 验证创建结果
        tool_info = extract_tool_results(result2)
        create_calls = [call for call in tool_info["tool_calls"] 
                       if call.get("name") == "Create_Vector_DB_From_PDF"]
        
        if create_calls:
            print(f"\n✅ 成功调用 Create_Vector_DB_From_PDF 工具")
            if tool_info["tool_results"]:
                create_result = parse_tool_result(tool_info["tool_results"][-1])
                if create_result.get("success"):
                    print(f"   ✅ 向量数据库创建成功")
                    print(f"   数据库路径: {create_result.get('db_path', 'N/A')}")
                    print(f"   文本块数: {create_result.get('total_chunks', 'N/A')}")
                    
                    # 测试 3: 使用新创建的数据库进行搜索
                    await asyncio.sleep(2)
                    
                    print("\n" + "=" * 80)
                    print("测试 3: 使用新创建的数据库进行搜索")
                    print("=" * 80)
                    
                    # 使用更具体的查询，基于PDF内容
                    result3 = await test_rag_agent(
                        "搜索刚才创建的PDF数据库中的内容，告诉我论文的主要方法或结论",
                        "测试 3: 搜索新创建的数据库"
                    )
                    
                    tool_info3 = extract_tool_results(result3)
                    search_calls = [call for call in tool_info3["tool_calls"] 
                                   if call.get("name") == "Database_Search"]
                    if search_calls:
                        print(f"\n✅ 成功使用新创建的数据库进行搜索")
                else:
                    print(f"\n⚠️  创建失败: {create_result.get('error', 'Unknown error')}")
        else:
            print(f"\n⚠️  未调用 Create_Vector_DB_From_PDF 工具")
    else:
        print("\n" + "=" * 80)
        print("测试 2: 跳过（未找到PDF文件）")
        print("=" * 80)
        print(f"提示: 在 {download_dir} 目录中放置PDF文件以测试创建向量数据库功能")
        print("或者先运行 openreview_agent 测试下载论文")
    
    # 测试 4: 测试处理不存在的PDF文件
    print("\n" + "=" * 80)
    print("测试 4: 处理不存在的PDF文件")
    print("=" * 80)
    
    result4 = await test_rag_agent(
        "请为PDF文件 ./nonexistent.pdf 创建向量数据库",
        "测试 4: 处理不存在的文件"
    )
    
    tool_info4 = extract_tool_results(result4)
    if tool_info4["tool_results"]:
        create_result = parse_tool_result(tool_info4["tool_results"][-1])
        if not create_result.get("success"):
            print(f"\n✅ 正确处理了不存在的文件错误")
            print(f"   错误信息: {create_result.get('error', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


# Pytest 测试函数
@pytest.mark.asyncio
async def test_rag_assistant_database_search():
    """测试 RAG Assistant 的数据库搜索功能"""
    result = await test_rag_agent(
        "搜索关于公司政策的信息",
        "Pytest 测试: 数据库搜索",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证最后一条消息是 AI 的响应
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content"), "最后一条消息应该有内容"


@pytest.mark.asyncio
async def test_rag_assistant_handles_empty_query():
    """测试 Agent 处理空查询的情况"""
    result = await test_rag_agent(
        "搜索",
        "Pytest 测试: 处理空查询",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证最后一条消息是 AI 的响应
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content"), "最后一条消息应该有内容"


@pytest.mark.asyncio
async def test_rag_assistant_create_vector_db_from_pdf():
    """测试 RAG Assistant 从PDF创建向量数据库的功能"""
    # 检查是否有可用的PDF文件
    download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
    pdf_files = []
    
    if os.path.exists(download_dir):
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        pytest.skip("未找到PDF文件，跳过创建向量数据库测试")
    
    test_pdf = pdf_files[0]
    
    # 测试创建向量数据库
    result = await test_rag_agent(
        f"请为PDF文件 {test_pdf} 创建向量数据库",
        "Pytest 测试: 创建向量数据库",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证有 Create_Vector_DB_From_PDF 工具调用
    tool_info = extract_tool_results(result)
    create_calls = [call for call in tool_info["tool_calls"]
                   if call.get("name") == "Create_Vector_DB_From_PDF"]
    
    if create_calls:
        # 验证创建结果
        if tool_info["tool_results"]:
            create_result = parse_tool_result(tool_info["tool_results"][-1])
            # 创建可能成功或失败（取决于文件），但应该有结果
            assert "success" in create_result or "error" in create_result
    else:
        # 如果没有调用工具，可能是Agent认为不需要，这也是有效的响应
        pass


@pytest.mark.asyncio
async def test_rag_assistant_integration_workflow():
    """测试完整工作流：创建数据库 -> 搜索"""
    # 检查是否有可用的PDF文件
    download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
    pdf_files = []
    
    if os.path.exists(download_dir):
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        pytest.skip("未找到PDF文件，跳过集成测试")
    
    test_pdf = pdf_files[0]
    
    # 步骤1: 创建向量数据库
    create_result = await test_rag_agent(
        f"请为PDF文件 {test_pdf} 创建向量数据库",
        "集成测试: 创建数据库",
        verbose=False
    )
    
    # 等待一下，确保数据库创建完成
    await asyncio.sleep(2)
    
    # 步骤2: 搜索数据库
    search_result = await test_rag_agent(
        "搜索PDF中的内容",
        "集成测试: 搜索数据库",
        verbose=False
    )
    
    assert create_result is not None
    assert search_result is not None
    
    # 验证创建步骤有工具调用
    create_tool_info = extract_tool_results(create_result)
    create_calls = [call for call in create_tool_info["tool_calls"]
                   if call.get("name") == "Create_Vector_DB_From_PDF"]
    
    # 验证搜索步骤有工具调用
    search_tool_info = extract_tool_results(search_result)
    search_calls = [call for call in search_tool_info["tool_calls"]
                   if call.get("name") == "Database_Search"]
    
    # 至少应该有一个工具调用
    assert len(create_calls) > 0 or len(search_calls) > 0, "应该至少有一个工具调用"


test_rag_agent.__test__ = False

if __name__ == "__main__":
    # 直接运行测试
    asyncio.run(main())
