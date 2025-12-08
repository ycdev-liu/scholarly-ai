from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    
    Uses CHECKPOINTER_DB_TYPE if set, otherwise falls back to DATABASE_TYPE for backward compatibility.
    """
    # Use CHECKPOINTER_DB_TYPE if explicitly set, otherwise use DATABASE_TYPE
    db_type = settings.CHECKPOINTER_DB_TYPE or settings.DATABASE_TYPE
    
    if db_type == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if db_type == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # Default to SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    
    Uses STORE_DB_TYPE if set, otherwise falls back to DATABASE_TYPE for backward compatibility.
    """
    # Use STORE_DB_TYPE if explicitly set, otherwise use DATABASE_TYPE
    db_type = settings.STORE_DB_TYPE or settings.DATABASE_TYPE
    
    if db_type == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    else:  # Default to SQLite (which uses InMemoryStore)
        return get_sqlite_store()


__all__ = ["initialize_database", "initialize_store"]
