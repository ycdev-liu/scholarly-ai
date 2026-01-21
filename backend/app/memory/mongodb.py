import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from core.settings import settings


def _has_auth_credentials() -> bool:
    required_auth = ["MONGO_USER", "MONGO_PASSWORD", "MONGO_AUTH_SOURCE"]
    set_auth = [var for var in required_auth if getattr(settings, var, None)]
    if len(set_auth) > 0 and len(set_auth) != len(required_auth):
        raise ValueError(
            f"If any of the following environment variables are set, all must be set: {', '.join(required_auth)}."
        )
    return len(set_auth) == len(required_auth)


def validate_mongo_config() -> None:
    """
    验证所有必需的MongoDB配置是否存在。
    如果缺少任何必需的配置，则抛出ValueError。
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {', '.join(missing_always)}. "
            "These environment variables must be set to use MongoDB persistence."
        )

    _has_auth_credentials()


def get_mongo_connection_string() -> str:
    """从设置中构建并返回MongoDB连接字符串。"""

    if _has_auth_credentials():
        if settings.MONGO_PASSWORD is None:  # 用于类型检查
            raise ValueError("MONGO_PASSWORD is not set")
        password = settings.MONGO_PASSWORD.get_secret_value().strip()
        password_escaped = urllib.parse.quote_plus(password)
        return (
            f"mongodb://{settings.MONGO_USER}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={settings.MONGO_AUTH_SOURCE}"
        )
    else:
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/"


def get_mongo_saver() -> AbstractAsyncContextManager[AsyncMongoDBSaver]:
    """初始化并返回MongoDB保存器实例。"""
    validate_mongo_config()
    if settings.MONGO_DB is None:  # 用于类型检查
        raise ValueError("MONGO_DB is not set")
    return AsyncMongoDBSaver.from_conn_string(
        get_mongo_connection_string(), db_name=settings.MONGO_DB
    )
