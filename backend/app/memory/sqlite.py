from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """初始化并返回SQLite保存器实例。"""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """为InMemoryStore提供异步上下文管理器接口的包装器。"""

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # InMemoryStore不需要清理
        pass

    async def setup(self):
        # 空操作方法，用于与PostgresStore兼容
        pass


@asynccontextmanager
async def get_sqlite_store():
    """初始化并返回用于长期记忆的存储实例。

    注意：LangGraph中没有SQLite特定的存储，
    因此我们使用包装在异步上下文管理器中的InMemoryStore以实现兼容性。
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()
