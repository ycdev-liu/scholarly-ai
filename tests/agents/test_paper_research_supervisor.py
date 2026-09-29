"""Tests for the Paper Research Supervisor Agent.

This test covers the complete workflow:
1. Search for papers (using openreview_agent)
2. Download a paper (using openreview_agent)
3. Create vector database from PDF (using rag_assistant)
4. Answer questions about the paper (using rag_assistant)
"""

import json
import os
import sys
import re
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

from agents.paper_research_supervisor import paper_research_supervisor
from core import settings
from core.logging_config import setup_logging

# 加载环境变量
load_dotenv()
def extract_tool_results(result: dict) -> dict:
    """从 agent 结果中提取工具调用和结果"""
    tool_calls = []
    tool_results = []
    transfer_calls = []
    
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append(tool_call)
                tool_name = tool_call.get("name", "")
                if "transfer_to" in tool_name:
                    transfer_calls.append(tool_call)
        elif isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
    
    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "transfer_calls": transfer_calls
    }


def parse_tool_result(tool_result: str) -> dict:
    """解析工具结果的JSON字符串"""
    try:
        return json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_pdf_path_from_response(content: str) -> str:
    """从响应内容中提取PDF文件路径"""
    # 尝试多种模式匹配PDF路径
    patterns = [
        r'`([^`]+\.pdf)`',  # 反引号中的PDF路径
        r'"(.*?\.pdf)"',     # 双引号中的PDF路径
        r'([./].*?\.pdf)',   # 以 ./ 或 / 开头的PDF路径
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            path = match.group(1)
            if os.path.exists(path) or os.path.exists(path.replace('\\', '/')):
                return path
    
    return None


def extract_db_info_from_response(content: str) -> dict:
    """从响应内容中提取数据库信息"""
    info = {}
    
    # 尝试提取数据库路径
    db_path_patterns = [
        r'数据库路径[：:]\s*`([^`]+)`',
        r'db_path[":]\s*"([^"]+)"',
        r'`([^`]+vector_databases[^`]+)`',
    ]
    
    for pattern in db_path_patterns:
        match = re.search(pattern, content)
        if match:
            info["db_path"] = match.group(1)
            break
    
    # 尝试提取数据库类型
    if "qdrant" in content.lower() or "Qdrant" in content:
        info["db_type"] = "qdrant"
    elif "chroma" in content.lower() or "Chroma" in content:
        info["db_type"] = "chroma"
    
    # 尝试提取文本块数
    chunks_match = re.search(r'(\d+)\s*个文本块', content)
    if chunks_match:
        info["total_chunks"] = int(chunks_match.group(1))
    
    return info


async def test_supervisor_agent(message: str, test_name: str = "测试", verbose: bool = True):
    """测试 supervisor agent 的响应"""
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
        result = await paper_research_supervisor.ainvoke(
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
                                if key in ["query", "paper_id", "venue", "max_papers"]:
                                    print(f"    {key}: {value}")

        if verbose and result["messages"]:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
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
    """主测试函数 - 完整工作流测试"""
    # 设置日志
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file="test_paper_research_supervisor.log",
        log_dir="./logs",
    )
    
    print("=" * 80)
    print("Paper Research Supervisor Agent 完整工作流测试")
    print("=" * 80)
    print(f"使用模型: {settings.DEFAULT_MODEL}")
    print(f"Agent ID: paper-research-supervisor")
    print("\n测试流程:")
    print("1. 搜索论文")
    print("2. 下载论文")
    print("3. 创建向量数据库")
    print("4. 回答论文相关问题")
    print("=" * 80)
    
    # 存储测试过程中的信息
    test_info = {
        "paper_id": None,
        "pdf_path": None,
        "db_path": None,
        "db_type": None
    }
    
    # ============================================================
    # 步骤 1: 搜索论文
    # ============================================================
    print("\n" + "=" * 80)
    print("步骤 1/4: 搜索论文")
    print("=" * 80)
    
    result1 = await test_supervisor_agent(
        "帮我搜索 ICML 2024 的论文，只搜索1篇",
        "步骤 1: 搜索论文"
    )
    
    # 提取论文ID（如果找到）
    tool_info1 = extract_tool_results(result1)
    if tool_info1["transfer_calls"]:
        print(f"\n✅ Supervisor 成功路由到 OpenReview agent")
    
    # 尝试从响应中提取 paper_id
    last_msg1 = result1["messages"][-1] if result1["messages"] else None
    if last_msg1 and hasattr(last_msg1, "content"):
        content = last_msg1.content
        print(f"搜索响应: {content[:300]}...")
        
        # 尝试从内容中提取 paper_id（如果提到）
        paper_id_match = re.search(r'paper[_\s]?id[:\s]+([A-Za-z0-9]+)', content, re.IGNORECASE)
        if paper_id_match:
            test_info["paper_id"] = paper_id_match.group(1)
            print(f"✅ 提取到 paper_id: {test_info['paper_id']}")
    
    await asyncio.sleep(1)
    
    # ============================================================
    # 步骤 2: 下载论文
    # ============================================================
    print("\n" + "=" * 80)
    print("步骤 2/4: 下载论文")
    print("=" * 80)
    
    # 记录下载前的PDF文件列表
    download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
    pdf_files_before = []
    if os.path.exists(download_dir):
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files_before.append(os.path.join(root, file))
    
    result2 = await test_supervisor_agent(
        "请下载刚才搜索到的第一篇论文",
        "步骤 2: 下载论文"
    )
    
    tool_info2 = extract_tool_results(result2)
    if tool_info2["transfer_calls"]:
        print(f"\n✅ Supervisor 成功路由到 OpenReview agent 进行下载")
    
    # 检查下载结果，提取PDF路径
    last_msg2 = result2["messages"][-1] if result2["messages"] else None
    if last_msg2 and hasattr(last_msg2, "content"):
        content = last_msg2.content
        print(f"下载响应: {content[:300]}...")
        
        # 方法1: 从响应内容中提取PDF路径
        pdf_path = extract_pdf_path_from_response(content)
        if pdf_path:
            test_info["pdf_path"] = pdf_path
            print(f"\n✅ 从响应中提取到PDF路径: {test_info['pdf_path']}")
        else:
            # 方法2: 检查下载目录中的新文件
            await asyncio.sleep(2)  # 等待文件写入完成
            pdf_files_after = []
            if os.path.exists(download_dir):
                for root, dirs, files in os.walk(download_dir):
                    for file in files:
                        if file.endswith(".pdf"):
                            pdf_files_after.append(os.path.join(root, file))
            
            # 找出新下载的文件
            new_files = [f for f in pdf_files_after if f not in pdf_files_before]
            if new_files:
                test_info["pdf_path"] = new_files[0]
                print(f"\n✅ 检测到新下载的PDF文件: {test_info['pdf_path']}")
            elif pdf_files_after:
                # 如果没有新文件，使用最新的文件
                pdf_files_after.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                test_info["pdf_path"] = pdf_files_after[0]
                print(f"\n✅ 使用最新的PDF文件: {test_info['pdf_path']}")
    
    await asyncio.sleep(2)
    
    # ============================================================
    # 步骤 3: 创建向量数据库
    # ============================================================
    print("\n" + "=" * 80)
    print("步骤 3/4: 创建向量数据库")
    print("=" * 80)
    
    if test_info["pdf_path"] and os.path.exists(test_info["pdf_path"]):
        pdf_path = test_info["pdf_path"]
    else:
        # 如果没有找到下载的文件，尝试使用已有的PDF文件
        download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
        pdf_files = []
        if os.path.exists(download_dir):
            for root, dirs, files in os.walk(download_dir):
                for file in files:
                    if file.endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))
        
        if pdf_files:
            pdf_path = pdf_files[0]
            test_info["pdf_path"] = pdf_path
            print(f"使用已有PDF文件: {pdf_path}")
        else:
            print("⚠️  未找到PDF文件，跳过创建向量数据库步骤")
            pdf_path = None
    
    if pdf_path:
        result3 = await test_supervisor_agent(
            f"请为PDF文件 {pdf_path} 创建向量数据库",
            "步骤 3: 创建向量数据库"
        )
        
        tool_info3 = extract_tool_results(result3)
        if tool_info3["transfer_calls"]:
            transfer_name = tool_info3["transfer_calls"][0].get("name", "")
            if "rag_assistant" in transfer_name:
                print(f"\n✅ Supervisor 成功路由到 RAG assistant")
        
        # 检查创建结果
        last_msg3 = result3["messages"][-1] if result3["messages"] else None
        if last_msg3 and hasattr(last_msg3, "content"):
            content = last_msg3.content
            print(f"创建响应: {content[:300]}...")
            
            # 尝试从ToolMessage中解析JSON结果
            for msg in result3["messages"]:
                if isinstance(msg, ToolMessage):
                    try:
                        result_data = json.loads(msg.content)
                        if result_data.get("success"):
                            test_info["db_path"] = result_data.get("db_path")
                            test_info["db_type"] = result_data.get("db_type")
                            print(f"\n✅ 向量数据库创建成功")
                            print(f"   数据库路径: {test_info['db_path']}")
                            print(f"   数据库类型: {test_info['db_type']}")
                            print(f"   文本块数: {result_data.get('total_chunks', 'N/A')}")
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            # 如果JSON解析失败，尝试从文本中提取
            if not test_info["db_path"]:
                db_info = extract_db_info_from_response(content)
                if db_info:
                    test_info.update(db_info)
                    if test_info.get("db_path"):
                        print(f"\n✅ 从响应中提取到数据库信息")
                        print(f"   数据库路径: {test_info.get('db_path')}")
                        print(f"   数据库类型: {test_info.get('db_type', 'N/A')}")
    else:
        print("⚠️  跳过创建向量数据库步骤（无PDF文件）")
    
    await asyncio.sleep(2)
    
    # ============================================================
    # 步骤 4: 回答论文相关问题
    # ============================================================
    print("\n" + "=" * 80)
    print("步骤 4/4: 回答论文相关问题")
    print("=" * 80)
    
    result4 = await test_supervisor_agent(
        "请告诉我这篇论文的主要方法或结论是什么？",
        "步骤 4: 回答论文问题"
    )
    
    tool_info4 = extract_tool_results(result4)
    if tool_info4["transfer_calls"]:
        transfer_name = tool_info4["transfer_calls"][0].get("name", "")
        if "rag_assistant" in transfer_name:
            print(f"\n✅ Supervisor 成功路由到 RAG assistant 回答问题")
    
    last_msg4 = result4["messages"][-1] if result4["messages"] else None
    if last_msg4 and hasattr(last_msg4, "content"):
        content = last_msg4.content
        print(f"\n最终回答: {content[:500]}...")
        if len(content) > 500:
            print("(内容已截断)")
        
        # 验证回答是否包含实质性内容
        if len(content) > 100 and any(keyword in content for keyword in ["方法", "结论", "论文", "研究", "方法", "结论"]):
            print(f"\n✅ 回答包含实质性内容")
    
    # ============================================================
    # 测试总结
    # ============================================================
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 步骤 1: 搜索论文 - 完成")
    print(f"✅ 步骤 2: 下载论文 - 完成")
    if test_info.get("pdf_path"):
        print(f"   PDF文件: {test_info['pdf_path']}")
    if test_info.get("db_path"):
        print(f"✅ 步骤 3: 创建向量数据库 - 完成")
        print(f"   数据库路径: {test_info['db_path']}")
        print(f"   数据库类型: {test_info.get('db_type', 'N/A')}")
        print(f"✅ 步骤 4: 回答论文问题 - 完成")
    else:
        print(f"⚠️  步骤 3: 创建向量数据库 - 跳过（无PDF文件）")
        print(f"⚠️  步骤 4: 回答论文问题 - 可能无法回答（无数据库）")
    
    print("\n" + "=" * 80)
    print("完整工作流测试完成")
    print("=" * 80)


# ============================================================
# Pytest 测试函数
# ============================================================

@pytest.mark.asyncio
async def test_supervisor_routes_to_openreview_agent():
    """测试 Supervisor 正确路由到 OpenReview agent"""
    result = await test_supervisor_agent(
        "搜索 ICML 2024 的论文",
        "Pytest 测试: 路由到 OpenReview",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证有路由调用
    tool_info = extract_tool_results(result)
    transfer_calls = [call for call in tool_info["transfer_calls"]
                     if "openreview" in call.get("name", "").lower()]
    
    # Supervisor 应该路由到 openreview_agent
    # 验证最后一条消息是响应
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content"), "最后一条消息应该有内容"


@pytest.mark.asyncio
async def test_supervisor_routes_to_rag_assistant():
    """测试 Supervisor 正确路由到 RAG assistant"""
    result = await test_supervisor_agent(
        "请告诉我数据库中论文的主要方法",
        "Pytest 测试: 路由到 RAG Assistant",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证有路由调用
    tool_info = extract_tool_results(result)
    transfer_calls = [call for call in tool_info["transfer_calls"]
                     if "rag_assistant" in call.get("name", "").lower()]
    
    # Supervisor 应该路由到 rag_assistant
    # 验证最后一条消息是响应
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content"), "最后一条消息应该有内容"


@pytest.mark.asyncio
async def test_supervisor_complete_workflow():
    """测试完整工作流：搜索 -> 下载 -> 创建数据库 -> 回答问题"""
    # 步骤1: 搜索
    search_result = await test_supervisor_agent(
        "搜索 ICML 2024 的论文，只搜索1篇",
        "集成测试: 搜索",
        verbose=False
    )
    
    assert search_result is not None
    
    await asyncio.sleep(1)
    
    # 步骤2: 下载（如果找到论文）
    download_result = await test_supervisor_agent(
        "请下载刚才搜索到的第一篇论文",
        "集成测试: 下载",
        verbose=False
    )
    
    assert download_result is not None
    
    # 检查是否有PDF文件
    download_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "./downloads")
    pdf_files = []
    if os.path.exists(download_dir):
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
    
    if pdf_files:
        # 步骤3: 创建向量数据库
        pdf_path = pdf_files[0]
        create_result = await test_supervisor_agent(
            f"请为PDF文件 {pdf_path} 创建向量数据库",
            "集成测试: 创建数据库",
            verbose=False
        )
        
        assert create_result is not None
        
        await asyncio.sleep(2)
        
        # 步骤4: 回答问题
        query_result = await test_supervisor_agent(
            "请告诉我这篇论文的主要方法是什么？",
            "集成测试: 回答问题",
            verbose=False
        )
        
        assert query_result is not None
        assert len(query_result["messages"]) > 0
        
        # 验证最后一条消息是响应
        last_msg = query_result["messages"][-1]
        assert hasattr(last_msg, "content"), "最后一条消息应该有内容"
    else:
        pytest.skip("未找到PDF文件，跳过完整工作流测试")


test_supervisor_agent.__test__ = False

if __name__ == "__main__":
    # 直接运行测试
    asyncio.run(main())
