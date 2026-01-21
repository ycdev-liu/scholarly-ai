from functools import cache
from typing import TypeAlias
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.settings import settings

# print(settings)
# print("=" * 60)
# print("模型加载完成")
# print("=" * 60)
# print(f"默认模型: {settings.DEFAULT_MODEL}")
# print(f"可用模型数量: {len(settings.AVAILABLE_MODELS)}")
# print(f"模式: {settings.MODE or '未设置'}")
# print(f"主机: {settings.HOST}:{settings.PORT}")
# if settings.COMPATIBLE_BASE_URL:
#     print(f"兼容模式 API: {settings.COMPATIBLE_BASE_URL}")
#     print(f"兼容模式模型: {settings.COMPATIBLE_MODEL}")
# print("=" * 60)
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


ModelT: TypeAlias = (
    ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatOllama
    | FakeToolModel
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # 注意：如果/stream端点以stream_tokens=True（默认值）调用，则streaming=True的模型将在生成时发送token
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
        
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            base_url=settings.COMPATIBLE_BASE_URL,
            api_key=settings.COMPATIBLE_API_KEY.get_secret_value() if settings.COMPATIBLE_API_KEY else None,
        )
    
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://api.deepseek.com",
            api_key=settings.DEEPSEEK_API_KEY.get_secret_value() if settings.DEEPSEEK_API_KEY else None,
        )
    
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
        
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")