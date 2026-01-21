from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.settings import settings


def validate_postgres_config() -> None:
    """
    验证所有必需的PostgreSQL配置是否存在。
    如果缺少任何必需的配置，则抛出ValueError。
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )

    if settings.POSTGRES_MIN_CONNECTIONS_PER_POOL > settings.POSTGRES_MAX_CONNECTIONS_PER_POOL:
        raise ValueError(
            f"POSTGRES_MIN_CONNECTIONS_PER_POOL ({settings.POSTGRES_MIN_CONNECTIONS_PER_POOL}) must be less than or equal to POSTGRES_MAX_CONNECTIONS_PER_POOL ({settings.POSTGRES_MAX_CONNECTIONS_PER_POOL})"
        )


def get_postgres_connection_string() -> str:
    """从设置中构建并返回PostgreSQL连接字符串。"""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


@asynccontextmanager
async def get_postgres_saver():
    """基于连接池初始化并返回PostgreSQL保存器实例，以获得更稳定的连接。"""
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph要求autocommit=true且row_factory设置为dict_row。
        # 传递application_name以便在Postgres数据库连接管理器中识别连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 确保在使用连接前连接仍然有效
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            yield checkpointer
        finally:
            await pool.close()


@asynccontextmanager
async def get_postgres_store():
    """
    基于连接池获取PostgreSQL存储实例，以获得更稳定的连接。

    返回一个可与异步上下文管理器模式一起使用的AsyncPostgresStore实例。

    """
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "store"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph要求autocommit=true且row_factory设置为dict_row
        # 传递application_name以便在Postgres数据库连接管理器中识别连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 确保在使用连接前连接仍然有效
        check=AsyncConnectionPool.check_connection,
    ) as pool: 
        try:
            store = AsyncPostgresStore(pool)
            await store.setup()
            yield store
        finally:
            await pool.close()
