import asyncio
from re import T
import sys
import uvicorn
from dotenv import load_dotenv
from core import settings

load_dotenv()

if __name__ == "__main__":
    # 服务器绑定使用 0.0.0.0 以接受所有网络接口的连接
    # 但客户端连接时 BASE_URL 会自动转换为 localhost
    host = settings.HOST if settings.HOST != "localhost" else "0.0.0.0"
    
    print(f"Starting server on {host}:{settings.PORT}")
    print(f"Client BASE_URL: {settings.BASE_URL}")
    
    uvicorn.run(
        "service:app",
        host=host,
        port=settings.PORT,
        reload=True,
        reload_excludes=["logs/*", "*.log", "*.db", "*.db-wal", "*.db-shm", ".git/*", "__pycache__/*"]
        # timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
