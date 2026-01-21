import inspect
import json
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from collections.abc import AsyncGenerator
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from typing import TYPE_CHECKING

from agents import DEFAULT_AGENT, get_agent
from core import settings
from schema import ChatMessage, StreamInput, UserInput

if TYPE_CHECKING:
    from agents import AgentGraph


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """从LangChain消息创建ChatMessage。"""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """从内容中移除工具调用。"""
    if isinstance(content, str):
        return content
    # 目前只有Anthropic模型流式传输工具调用，使用内容项类型tool_use。
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


async def _handle_input(user_input: UserInput, agent: "AgentGraph") -> tuple[dict[str, Any], UUID]:
    """
    解析用户输入并处理任何需要的中断恢复。
    返回用于代理调用的kwargs和run_id。
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    user_id = user_input.user_id or str(uuid4())
    

    configurable = {"thread_id": thread_id, "user_id": user_id}
    # 如果用户输入的模型不为空，则设置模型
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks = []
    # 如果开启了Langfuse追踪，则添加Langfuse回调
    if settings.LANGFUSE_TRACING:
        # 为Langchain初始化Langfuse CallbackHandler（追踪）
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    # 如果用户输入的agent_config不为空，则更新configurable
    if user_input.agent_config:
        # 检查保留键（包括'model'，即使不在configurable中）
        reserved_keys = {"thread_id", "user_id", "model"}
        # 如果用户输入的agent_config包含保留关键字，则抛出错误
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    # 创建RunnableConfig
    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # 检查需要恢复的中断
    # 获取状态
    state = await agent.aget_state(config=config)
    # 获取中断任务
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]
    # 如果存在中断任务，则创建命令
    input: Command | dict[str, Any]
    if interrupted_tasks:
        # 假设用户输入是对中断恢复代理执行的响应
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id



async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:  # type: ignore[type-arg]
    """
    从代理生成消息流。
    这是/stream端点的主要方法。
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        # 处理来自图的流式事件，并通过SSE流产生消息。
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # 基于子图处理不同的流事件结构
            if len(stream_event) == 3:
                # 使用subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # 不使用子图: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # 使用langgraph-supervisor库的特殊情况
                    if "supervisor" in node or "sub-agent" in node:
                        # 来自实际代理的唯一工具是handoff和handback工具
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # 如果这是子代理，我们希望保留最后2条消息 - handback工具及其结果
                                update_messages = update_messages[-2:]
                            else:
                                # 如果这是监督者，我们只希望保留最后一条消息 - handoff结果。工具来自'agent'节点。
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # 将部分存储在临时字典中
                    current_message[key] = value
                else:
                    # 如果正在进行中，添加完整消息
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # 添加任何剩余的消息部分
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph重新发送输入消息，这感觉很奇怪，所以丢弃它
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # 在OpenAI的上下文中，空内容通常意味着
                    # 模型正在请求调用工具。
                    # 所以我们只打印非空内容。
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"




def _create_ai_message(parts: dict) -> AIMessage:
    """从部分字典创建 AIMessage"""
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    """返回 SSE 响应示例"""
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    """验证 Bearer Token 认证"""
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)