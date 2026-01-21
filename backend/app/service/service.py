"""FastAPI 服务主文件"""
import warnings
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.routing import APIRoute
from langchain_core._api import LangChainBetaWarning

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info, load_agent
from core import settings
from memory import initialize_database, initialize_store
from schema import ServiceMetadata
from service.routers import agent, feedback, history, metadata, vectordb,document

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def custom_generate_unique_id(route: APIRoute) -> str:
    """为OpenAPI客户端生成生成惯用的操作ID。"""
    return route.name

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    可配置的生命周期，初始化适当的数据库检查点、存储
    和代理，支持异步加载 - 例如用于启动MCP客户端。
    """
    try:
        # 初始化检查点（用于短期记忆）和存储（用于长期记忆）
        async with initialize_database() as saver, initialize_store() as store:
            # 设置两个组件
            logger.info("Initializing database and store")
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()

            # 仅为Postgres设置存储，因为InMemoryStore不需要设置
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # 使用两个内存组件和异步加载配置代理
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                except Exception as e:
                    # 继续处理其他代理，而不是使启动失败
                    pass

                agent = get_agent(a.key)
                # 为线程作用域内存（对话历史）设置检查点
                agent.checkpointer = saver
                # 为长期记忆（跨对话知识）设置存储
                agent.store = store

            # 在配置代理后测试数据库连接
            try:
                from langchain_core.runnables import RunnableConfig

                test_config = RunnableConfig(configurable={"thread_id": "__init_test__"})
                # 尝试获取状态（这将测试数据库连接）
                test_agent = get_agent(DEFAULT_AGENT)
                if hasattr(test_agent, "aget_state"):
                    test_state = await test_agent.aget_state(config=test_config)
            except Exception as e:
                pass
            yield
    except Exception as e:
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)

# 注册路由（先注册公开路由，再注册需要认证的路由）
app.include_router(metadata.public_router)  # 公开路由：/api/info, /health
app.include_router(agent.router)  # /api/agents/*
app.include_router(feedback.router)  # /api/feedback
app.include_router(history.router)  # /api/history
app.include_router(vectordb.router)  # /api/vectordb/*
app.include_router(document.router)  # /api/document

# 为了向后兼容，保留旧的 /info 端点（重定向到新端点）
@app.get("/info")
async def info_legacy() -> ServiceMetadata:
    """向后兼容的 /info 端点"""
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    default_model = settings.DEFAULT_MODEL
    if default_model is None:
        # 如果没有设置默认模型，使用第一个可用模型
        default_model = models[0] if models else None
        if default_model is None:
            raise ValueError("No models available")
    
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=default_model,
    )
