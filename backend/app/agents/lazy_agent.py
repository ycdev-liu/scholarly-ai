"""具有异步初始化和动态图创建的代理类型。"""

from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


class LazyLoadingAgent(ABC):
    """需要异步加载的代理的基类。"""

    def __init__(self) -> None:
        """初始化代理。"""
        self._loaded = False
        self._graph: CompiledStateGraph | Pregel | None = None

    @abstractmethod
    async def load(self) -> None:
        """
        为此代理执行异步加载。
        此方法在服务启动期间调用，应处理：
        - 设置外部连接（MCP客户端、数据库等）
        - 加载工具或资源
        - 任何其他所需的异步设置
        - 创建代理的图
        """
        raise NotImplementedError  # pragma: no cover

    def get_graph(self) -> CompiledStateGraph | Pregel:
        """
        获取代理的图。

        返回在load()期间创建的图实例。

        Returns:
            代理的图（CompiledStateGraph或Pregel）
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")
        if self._graph is None:
            raise RuntimeError("Agent graph not created during load().")
        return self._graph
