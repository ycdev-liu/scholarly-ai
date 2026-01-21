"""代理调用相关的 API 路由"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agents import DEFAULT_AGENT, get_agent
from schema import ChatMessage, StreamInput, UserInput
from service.utils import (
    _handle_input,
    _sse_response_example,
    langchain_to_chat_message,
    message_generator,
    verify_bearer,
)

router = APIRouter(prefix="/api/agents", dependencies=[Depends(verify_bearer)])


@router.post("/invoke", operation_id="invoke_agent")
@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    使用用户输入调用代理以获取最终响应。

    如果未提供agent_id，将使用默认代理。
    使用thread_id来持久化并继续多轮对话。run_id kwarg
    也会附加到消息上以记录反馈。
    使用user_id来持久化并跨多个线程继续对话。
    """
    agent = get_agent(agent_id)
    # 处理输入
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # 正常响应，代理成功完成
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # 最后发生的是中断
            # 将第一个中断的值作为AIMessage返回
            from langchain_core.messages import AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_agent",
)
@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    流式传输代理对用户输入的响应，包括中间消息和token。

    如果未提供agent_id，将使用默认代理。
    使用thread_id来持久化并继续多轮对话。run_id kwarg
    也会附加到所有消息上以记录反馈。
    使用user_id来持久化并跨多个线程继续对话。

    设置`stream_tokens=false`以返回中间消息但不逐token返回。
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )
