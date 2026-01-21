"""历史记录相关的 API 路由"""
from fastapi import APIRouter, Depends, HTTPException

from agents import DEFAULT_AGENT, get_agent
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from schema import ChatHistory, ChatHistoryInput, ChatMessage
from service.utils import langchain_to_chat_message, verify_bearer

router = APIRouter(prefix="/api/history", dependencies=[Depends(verify_bearer)])


@router.post("", operation_id="get_chat_history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    获取聊天历史。
    """
    # TODO: 这里硬编码DEFAULT_AGENT有点奇怪
    agent = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unexpected error")
