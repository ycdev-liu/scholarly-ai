"""元数据和健康检查相关的 API 路由"""
from fastapi import APIRouter, Depends

from agents import DEFAULT_AGENT, get_all_agent_info
from core import settings
from core.settings import DatabaseType
from langchain_core.runnables import RunnableConfig
from pathlib import Path
from schema import ServiceMetadata
from service.utils import verify_bearer

import logging
logger = logging.getLogger(__name__)


# 公开路由（不需要认证）
public_router = APIRouter()


@public_router.get("/api/info", operation_id="get_service_info")
async def info() -> ServiceMetadata:
    """获取服务元数据，包括可用的代理和模型"""
    logger.info("Getting service info")
    try:
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
    except Exception as e:
        logger.error("Get/api/info error: %s", e)
        raise


@public_router.get("/health", operation_id="health_check")
async def health_check():
    """健康检查端点"""
    from agents import get_agent
    from typing import Any

    health_status: dict[str, Any] = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            from langfuse import Langfuse  # type: ignore[import-untyped]
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            health_status["langfuse"] = "disconnected"

    # 检查数据库状态（检查点和存储分别检查）
    try:
        db_info: dict[str, Any] = {}

        # 获取实际使用的数据库类型
        checkpointer_type = settings.CHECKPOINTER_DB_TYPE or settings.DATABASE_TYPE
        store_type = settings.STORE_DB_TYPE or settings.DATABASE_TYPE

        # 检查点状态（短期记忆）
        checkpointer_info: dict[str, Any] = {
            "type": checkpointer_type.value,
            "purpose": "short-term memory (conversation history)",
        }

        if checkpointer_type == DatabaseType.SQLITE:
            db_path = Path(settings.SQLITE_DB_PATH)
            checkpointer_info["file_exists"] = str(db_path.exists())
            if db_path.exists():
                checkpointer_info["file_size"] = f"{db_path.stat().st_size} bytes"
                checkpointer_info["file_path"] = str(db_path)

        # 测试检查点连接
        try:
            test_agent = get_agent(DEFAULT_AGENT)
            if hasattr(test_agent, "checkpointer") and getattr(test_agent, "checkpointer", None):
                test_config = RunnableConfig(configurable={"thread_id": "__health_check__"})
                if hasattr(test_agent, "aget_state"):
                    await test_agent.aget_state(config=test_config)  # type: ignore[attr-defined]
                    checkpointer_info["connection"] = "ok"
                else:
                    checkpointer_info["connection"] = "not_supported"
            else:
                checkpointer_info["connection"] = "not_configured"
        except Exception as e:
            checkpointer_info["connection"] = f"error: {str(e)[:50]}"

        db_info["checkpointer"] = checkpointer_info

        # 存储状态（长期记忆）
        store_info: dict[str, Any] = {
            "type": store_type.value,
            "purpose": "long-term memory (cross-conversation knowledge)",
        }

        if store_type == DatabaseType.POSTGRES:
            store_info["host"] = str(settings.POSTGRES_HOST or "not_configured")
            store_info["database"] = str(settings.POSTGRES_DB or "not_configured")
            # 测试 PostgreSQL 连接（如果已配置）
            if settings.POSTGRES_HOST and settings.POSTGRES_DB:
                try:
                    test_agent = get_agent(DEFAULT_AGENT)
                    if hasattr(test_agent, "store") and getattr(test_agent, "store", None):
                        store_info["connection"] = "ok"
                    else:
                        store_info["connection"] = "not_configured"
                except Exception as e:
                    store_info["connection"] = f"error: {str(e)[:50]}"
            else:
                store_info["connection"] = "not_configured"
        elif store_type == DatabaseType.SQLITE:
            store_info["note"] = "Using InMemoryStore (data not persisted)"
            store_info["connection"] = "ok"
        else:
            store_info["connection"] = "unknown"

        db_info["store"] = store_info

        health_status["database"] = db_info
    except Exception as e:
        health_status["database"] = {"error": str(e)[:100]}

    return health_status


# 需要认证的路由（如果需要）
router = APIRouter(prefix="/api/metadata", dependencies=[Depends(verify_bearer)])
