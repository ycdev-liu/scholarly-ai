from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field


class CustomData(BaseModel):
    "代理发送的自定义数据"

    data: dict[str, Any] = Field(description="自定义数据")

    def to_langchain(self) -> ChatMessage:
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())
