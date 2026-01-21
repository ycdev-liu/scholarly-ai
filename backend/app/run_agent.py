import argparse
import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
load_dotenv()

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info  # noqa: E402
from core import settings  # noqa: E402

async def main(agent_id: str, message: str, model: str | None = None) -> None:
    """运行指定的智能体"""


    try:
        #cast 类型检查 
        agent = cast(CompiledStateGraph, get_agent(agent_id))
    except (KeyError, RuntimeError) as e:
        print(f"错误: 无法加载智能体 '{agent_id}': {e}")
        print("\n可用的智能体:")
        for agent_info in get_all_agent_info():
            print(f"  - {agent_info.key}: {agent_info.description}")
        return

    # 确定使用的模型
    from schema.models import AllModelEnum  # noqa: E402
    
    if model:
        # 尝试将字符串转换为模型枚举
        try:
            # 尝试直接匹配
            model_enum = None
            for available_model in settings.AVAILABLE_MODELS:
                if str(available_model) == model or available_model.value == model:
                    model_enum = available_model
                    break
            
            if model_enum is None:
                print(f"警告: 模型 '{model}' 不在可用模型列表中")
                print(f"可用模型: {', '.join(sorted([str(m) for m in settings.AVAILABLE_MODELS]))}")
                print(f"使用默认模型: {settings.DEFAULT_MODEL}")
                model_to_use = settings.DEFAULT_MODEL
            else:
                model_to_use = model_enum
        except Exception as e:
            print(f"警告: 解析模型 '{model}' 时出错: {e}")
            print(f"使用默认模型: {settings.DEFAULT_MODEL}")
            model_to_use = settings.DEFAULT_MODEL
    else:
        model_to_use = settings.DEFAULT_MODEL

    print(f"使用智能体: {agent_id}")
    print(f"使用模型: {model_to_use}")
    print(f"查询: {message}\n")

    inputs: MessagesState = {
        "messages": [HumanMessage(message)]
    }

    config_dict = {"thread_id": str(uuid4())}
    if model_to_use:
        config_dict["model"] = model_to_use

    result = await agent.ainvoke(
        input=inputs,
        config=RunnableConfig(configurable=config_dict),
    )

    print("\n" + "=" * 80)
    print("智能体响应:")
    print("=" * 80)
    last_msg = result["messages"][-1]
    try:
        # 尝试使用 pretty_print
        if hasattr(last_msg, "pretty_print"):
            last_msg.pretty_print()
        else:
            print(last_msg.content)
    except UnicodeEncodeError:
        # 如果编码失败，直接打印内容
        print(last_msg.content)

    # 显示消息历史（用于调试）
    print(f"\n总共 {len(result['messages'])} 条消息")
    for idx, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        print(f"  消息 {idx + 1}: {msg_type}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"    工具调用: {len(msg.tool_calls)} 个")
            for tool_call in msg.tool_calls:
                print(f"      - {tool_call.get('name', 'unknown')}")

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 AI 智能体")
    parser.add_argument(
        "--agent",
        type=str,
        default="openreview-agent",
        help=f"要使用的智能体 ID (默认: openreview-agent)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="帮我搜索 ICML 2025 的论文",
        help="发送给智能体的消息 (默认: 搜索 ICML 2025 论文)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="要使用的模型 (默认: 使用 settings.DEFAULT_MODEL)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可用的模型",
    )

    args = parser.parse_args()

    if args.list_models:
        print("可用的模型:")
        for model in sorted(settings.AVAILABLE_MODELS):
            marker = " (默认)" if model == settings.DEFAULT_MODEL else ""
            print(f"  - {model}{marker}")
        print(f"\n默认模型: {settings.DEFAULT_MODEL}")
    else:
        asyncio.run(main(args.agent, args.message, args.model))
