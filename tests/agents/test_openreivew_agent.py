"""Tests for the OpenReview Agent."""

import json
import sys
from pathlib import Path
from uuid import uuid4

# 添加 src 目录到 Python 路径，以便直接运行测试文件时也能正确导入
if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import asyncio
import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agents.openreview_agent import openreview_agent
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


def parse_search_result(tool_result: str) -> dict:
    """解析搜索工具的结果"""
    try:
        return json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return {}


async def test_agent(message: str, test_name: str = "测试", verbose: bool = True):
    """测试 agent 的响应"""
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
        result = await openreview_agent.ainvoke(
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
                                if key in ["paper_id", "venue", "download_dir", "max_papers"]:
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
    """主测试函数"""
    # 设置日志
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file="test_openreview_agent.log",
        log_dir="./logs",
    )
    
    print("=" * 80)
    print("OpenReview Agent 测试")
    print("=" * 80)
    print(f"使用模型: {settings.DEFAULT_MODEL}")
    print(f"Agent ID: openreview-agent")
    
    # 测试 1: 搜索论文（使用更可靠的参数）
    print("\n" + "=" * 80)
    print("测试 1: 搜索 ICML 2024 的论文（使用更可靠的参数）")
    print("=" * 80)
    result1 = await test_agent(
        "帮我搜索 ICML 2024 的论文，只搜索3篇",
        "测试 1: 搜索论文"
    )
    
    # 验证搜索结果
    tool_info = extract_tool_results(result1)
    if tool_info["tool_results"]:
        search_data = parse_search_result(tool_info["tool_results"][0])
        if search_data.get("total_papers", 0) > 0:
            print(f"\n✅ 成功找到 {search_data['total_papers']} 篇论文")
            if search_data.get("papers"):
                first_paper = search_data["papers"][0]
                print(f"   第一篇论文: {first_paper.get('title', 'N/A')}")
                print(f"   Paper ID: {first_paper.get('id', 'N/A')}")
        else:
            print("\n⚠️  未找到论文，可能是搜索参数需要调整")
    
    # 等待一下，避免 API 限流
    print("\n等待 2 秒...")
    await asyncio.sleep(2)
    
    # 测试 2: 搜索 NeurIPS 2024（另一个会议）
    print("\n" + "=" * 80)
    print("测试 2: 搜索 NeurIPS 2024 的论文")
    print("=" * 80)
    result2 = await test_agent(
        "帮我搜索 NeurIPS 2024 的论文，只搜索2篇",
        "测试 2: 搜索 NeurIPS 论文"
    )
    
    await asyncio.sleep(2)
    
    # 测试 3: 搜索并下载论文（如果找到了论文）
    print("\n" + "=" * 80)
    print("测试 3: 搜索并下载论文")
    print("=" * 80)
    
    # 先搜索，获取一个 paper_id
    search_result = await test_agent(
        "搜索 ICML 2024 oral 的论文，只搜索1篇",
        "测试 3a: 先搜索论文",
        verbose=False
    )
    
    paper_id = None
    tool_info = extract_tool_results(search_result)
    if tool_info["tool_results"]:
        search_data = parse_search_result(tool_info["tool_results"][0])
        if search_data.get("papers") and len(search_data["papers"]) > 0:
            paper_id = search_data["papers"][0].get("id")
    
    if paper_id:
        print(f"找到论文 ID: {paper_id}")
        await asyncio.sleep(1)
        download_result = await test_agent(
            f"请下载 paper_id 为 {paper_id} 的论文",
            "测试 3b: 下载论文"
        )
        
        # 验证下载结果
        download_tool_info = extract_tool_results(download_result)
        if download_tool_info["tool_results"]:
            download_data = parse_search_result(download_tool_info["tool_results"][-1])
            if download_data.get("success"):
                print(f"\n✅ 下载成功！")
                print(f"   文件路径: {download_data.get('file_path', 'N/A')}")
                print(f"   文件大小: {download_data.get('file_size_kb', 'N/A')} KB")
            else:
                print(f"\n⚠️  下载失败: {download_data.get('error', 'Unknown error')}")
    else:
        print("\n⚠️  未找到论文，跳过下载测试")
        # 仍然测试下载功能（即使没有有效的 paper_id）
        await test_agent(
            "请尝试下载一篇论文",
            "测试 3b: 尝试下载（可能失败）"
        )
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


# Pytest 测试函数
@pytest.mark.asyncio
async def test_openreview_agent_search():
    """测试 OpenReview Agent 的搜索功能"""
    result = await test_agent(
        "帮我搜索 ICML 2024 的论文，只搜索3篇",
        "Pytest 测试: 搜索论文",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证至少有一个工具调用
    tool_info = extract_tool_results(result)
    assert len(tool_info["tool_calls"]) > 0, "应该至少有一个工具调用"
    
    # 验证有 OpenReview_Search 工具调用
    search_calls = [call for call in tool_info["tool_calls"] 
                   if call.get("name") == "OpenReview_Search"]
    assert len(search_calls) > 0, "应该有 OpenReview_Search 工具调用"


@pytest.mark.asyncio
async def test_openreview_agent_handles_no_results():
    """测试 Agent 处理无搜索结果的情况"""
    result = await test_agent(
        "搜索一个不存在的会议论文",
        "Pytest 测试: 处理无结果",
        verbose=False
    )
    
    assert result is not None
    assert len(result["messages"]) > 0
    
    # 验证最后一条消息是 AI 的响应
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content"), "最后一条消息应该有内容"


@pytest.mark.asyncio
async def test_openreview_agent_download():
    """测试 OpenReview Agent 的下载功能（如果找到论文）"""
    # 先搜索
    search_result = await test_agent(
        "搜索 ICML 2024 oral 的论文，只搜索1篇",
        "Pytest 测试: 搜索论文（用于下载）",
        verbose=False
    )
    
    # 提取 paper_id
    tool_info = extract_tool_results(search_result)
    paper_id = None
    
    if tool_info["tool_results"]:
        search_data = parse_search_result(tool_info["tool_results"][0])
        if search_data.get("papers") and len(search_data["papers"]) > 0:
            paper_id = search_data["papers"][0].get("id")
    
    if paper_id:
        # 测试下载
        download_result = await test_agent(
            f"请下载 paper_id 为 {paper_id} 的论文",
            "Pytest 测试: 下载论文",
            verbose=False
        )
        
        assert download_result is not None
        assert len(download_result["messages"]) > 0
        
        # 验证有 Download_Paper 工具调用
        download_tool_info = extract_tool_results(download_result)
        download_calls = [call for call in download_tool_info["tool_calls"]
                         if call.get("name") == "Download_Paper"]
        assert len(download_calls) > 0, "应该有 Download_Paper 工具调用"
    else:
        pytest.skip("未找到论文，跳过下载测试")


test_agent.__test__ = False

if __name__ == "__main__":
    # 直接运行测试
    from langchain_core.messages import HumanMessage
    asyncio.run(main())
