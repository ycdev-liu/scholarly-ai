from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings

from pathlib import Path


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    db_path = Path(settings.SQLITE_DB_PATH)
    
    # 确保目录存在
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    
    # 创建父目录（如果不存在）
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    return AsyncSqliteSaver.from_conn_string(str(db_path))


class AsyncInMemoryStore:

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for InMemoryStore
        pass

    async def setup(self):
        # No-op method for compatibility with PostgresStore
        pass


@asynccontextmanager
async def get_sqlite_store():
    """Initialize and return a store instance for long-term memory.

    Note: SQLite-specific store isn't available in LangGraph,
    so we use InMemoryStore wrapped in an async context manager for compatibility.
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()
