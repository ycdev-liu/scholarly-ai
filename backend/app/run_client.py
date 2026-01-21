import asyncio
import sys

from client import AgentClient, AgentClientError
from core import settings
from schema import ChatMessage


def print_help():
    """打印帮助信息"""
    print("\n" + "="*60)
    print("可用命令:")
    print("  /help, /h          - 显示帮助信息")
    print("  /agent <name>      - 切换 Agent")
    print("  /list              - 列出所有可用的 Agent")
    print("  /stream            - 切换流式/非流式模式")
    print("  /model <name>      - 设置模型（留空使用默认）")
    print("  /exit, /quit, /q   - 退出程序")
    print("="*60 + "\n")


async def async_interactive_mode() -> None:
    """异步交互模式"""
    print("="*60)
    print("启动异步交互模式")
    print("="*60)
    
    client = AgentClient(settings.BASE_URL)
    model = None
    use_streaming = True
    thread_id = None
    
    print(f"\n已连接到服务: {settings.BASE_URL}")
    print(f"当前 Agent: {client.agent}")
    print(f"流式模式: {'开启' if use_streaming else '关闭'}")
    print_help()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if command in ["/exit", "/quit", "/q"]:
                    print("再见！")
                    break
                elif command in ["/help", "/h"]:
                    print_help()
                elif command == "/list":
                    if client.info:
                        print("\n可用 Agent:")
                        for agent in client.info.agents:
                            marker = " (当前)" if agent.key == client.agent else ""
                            print(f"  - {agent.key}: {agent.description}{marker}")
                    print()
                elif command == "/agent":
                    if not arg:
                        print(f"当前 Agent: {client.agent}")
                    else:
                        try:
                            client.update_agent(arg)
                            print(f"已切换到 Agent: {client.agent}")
                        except AgentClientError as e:
                            print(f"错误: {e}")
                elif command == "/stream":
                    use_streaming = not use_streaming
                    print(f"流式模式: {'开启' if use_streaming else '关闭'}")
                elif command == "/model":
                    if not arg:
                        model = None
                        print("已重置为默认模型")
                    else:
                        model = arg
                        print(f"已设置模型: {model}")
                else:
                    print(f"未知命令: {command}。输入 /help 查看帮助")
                continue
            
            # 处理用户消息
            print("\nAgent: ", end="", flush=True)
            
            try:
                if use_streaming:
                    async for message in client.astream(
                        user_input, 
                        model=model,
                        thread_id=thread_id
                    ):
                        if isinstance(message, str):
                            print(message, end="", flush=True)
                        elif isinstance(message, ChatMessage):
                            if message.run_id:
                                thread_id = thread_id or str(message.run_id)
                            print()  # 换行
                else:
                    response = await client.ainvoke(
                        user_input,
                        model=model,
                        thread_id=thread_id
                    )
                    if response.run_id:
                        thread_id = thread_id or str(response.run_id)
                    print(response.content)
            except AgentClientError as e:
                print(f"\n错误: {e}")
            
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break


def sync_interactive_mode() -> None:
    """同步交互模式"""
    print("="*60)
    print("启动同步交互模式")
    print("="*60)
    print(settings.BASE_URL)
    
    client = AgentClient(settings.BASE_URL)
    model = None
    use_streaming = True
    thread_id = None
    
    print(f"\n已连接到服务: {settings.BASE_URL}")
    print(f"当前 Agent: {client.agent}")
    print(f"流式模式: {'开启' if use_streaming else '关闭'}")
    print_help()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            # 处理命令
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if command in ["/exit", "/quit", "/q"]:
                    print("再见！")
                    break
                elif command in ["/help", "/h"]:
                    print_help()
                elif command == "/list":
                    if client.info:
                        print("\n可用 Agent:")
                        for agent in client.info.agents:
                            marker = " (当前)" if agent.key == client.agent else ""
                            print(f"  - {agent.key}: {agent.description}{marker}")
                    print()
                elif command == "/agent":
                    if not arg:
                        print(f"当前 Agent: {client.agent}")
                    else:
                        try:
                            client.update_agent(arg)
                            print(f"已切换到 Agent: {client.agent}")
                        except AgentClientError as e:
                            print(f"错误: {e}")
                elif command == "/stream":
                    use_streaming = not use_streaming
                    print(f"流式模式: {'开启' if use_streaming else '关闭'}")
                elif command == "/model":
                    if not arg:
                        model = None
                        print("已重置为默认模型")
                    else:
                        model = arg
                        print(f"已设置模型: {model}")
                else:
                    print(f"未知命令: {command}。输入 /help 查看帮助")
                continue
            
            # 处理用户消息
            print("\nAgent: ", end="", flush=True)
            
            try:
                if use_streaming:
                    for message in client.stream(
                        user_input,
                        model=model,
                        thread_id=thread_id
                    ):
                        if isinstance(message, str):
                            print(message, end="", flush=True)
                        elif isinstance(message, ChatMessage):
                            if message.run_id:
                                thread_id = thread_id or str(message.run_id)
                            print()  # 换行
                else:
                    response = client.invoke(
                        user_input,
                        model=model,
                        thread_id=thread_id
                    )
                    if response.run_id:
                        thread_id = thread_id or str(response.run_id)
                    print(response.content)
            except AgentClientError as e:
                print(f"\n错误: {e}")
            
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break


if __name__ == "__main__":
    # 检查命令行参数
    mode = "sync"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "async":
        asyncio.run(async_interactive_mode())
    else:
        sync_interactive_mode()
