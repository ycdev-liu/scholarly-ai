import asyncio
import logging
from re import T
import sys
import uvicorn
from dotenv import load_dotenv
from core import settings
from core import setup_logging

load_dotenv()

if __name__ == "__main__":

    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        reload_excludes=["logs/*", "*.log", "*.db", "*.db-wal", "*.db-shm", ".git/*", "__pycache__/*"]
        # timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
