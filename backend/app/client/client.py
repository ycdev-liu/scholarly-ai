import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, List, Optional

from httpcore._sync import http2
import httpx


# os.environ["HTTP_PROXY"] = ""
# os.environ["HTTPS_PROXY"] = ""
# os.environ["http_proxy"] = ""
# os.environ["https_proxy"] = ""
# os.environ["NO_PROXY"] = "*"  

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """用于与代理服务交互的客户端。"""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        初始化客户端。

        Args:
            base_url (str): 代理服务的基础URL。
            agent (str): 要使用的默认代理名称。
            timeout (float, optional): 请求的超时时间。
            get_info (bool, optional): 是否在初始化时获取代理信息。
                默认值: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            # print(f"Retrieving info from {self.base_url}/info")
            # print("headers: ", self._headers)
            # print("timeout: ", self.timeout)
            # response = httpx.get(
            #     f"{self.base_url}/info",
            #     headers=self._headers,
            #     timeout=self.timeout,
            # )
            # response.raise_for_status()
            info_url = f"{self.base_url}/api/info"
            print(f"Retrieving info from {info_url}")
            
            # 准备 headers，添加 accept header
            headers = self._headers.copy()
            headers["accept"] = "application/json"
            
            print("headers: ", headers)
            print("timeout: ", self.timeout)

  
            
            # 使用 Client 来禁用 HTTP/2，避免 502 错误
            with httpx.Client(
                http2=False,  # 禁用 HTTP/2，强制使用 HTTP/1.1
                timeout=self.timeout or 30.0,  # 设置默认超时
                proxies=None,#禁用代理
            ) as client:
                response = client.get(
                    info_url,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        异步调用代理。仅返回最终消息。

        Args:
            message (str): 发送给代理的消息
            model (str, optional): 用于代理的LLM模型
            thread_id (str, optional): 用于继续对话的线程ID
            user_id (str, optional): 用于跨多个线程继续对话的用户ID
            agent_config (dict[str, Any], optional): 传递给代理的额外配置

        Returns:
            AnyMessage: 来自代理的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/agents/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        同步调用代理。仅返回最终消息。

        Args:
            message (str): 发送给代理的消息
            model (str, optional): 用于代理的LLM模型
            thread_id (str, optional): 用于继续对话的线程ID
            user_id (str, optional): 用于跨多个线程继续对话的用户ID
            agent_config (dict[str, Any], optional): 传递给代理的额外配置

        Returns:
            ChatMessage: 来自代理的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/api/agents/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # 将JSON格式的消息转换为AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # 直接产生字符串token
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        同步流式传输代理的响应。

        代理过程的每个中间消息都会作为ChatMessage产生。
        如果stream_tokens为True（默认值），响应还会在生成时产生来自流式模型的内容token。

        Args:
            message (str): 发送给代理的消息
            model (str, optional): 用于代理的LLM模型
            thread_id (str, optional): 用于继续对话的线程ID
            user_id (str, optional): 用于跨多个线程继续对话的用户ID
            agent_config (dict[str, Any], optional): 传递给代理的额外配置
            stream_tokens (bool, optional): 在生成时流式传输token
                默认值: True

        Returns:
            Generator[ChatMessage | str, None, None]: 来自代理的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/agents/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        异步流式传输代理的响应。

        代理过程的每个中间消息都会作为AnyMessage产生。
        如果stream_tokens为True（默认值），响应还会在生成时产生来自流式模型的内容token。

        Args:
            message (str): 发送给代理的消息
            model (str, optional): 用于代理的LLM模型
            thread_id (str, optional): 用于继续对话的线程ID
            user_id (str, optional): 用于跨多个线程继续对话的用户ID
            agent_config (dict[str, Any], optional): 传递给代理的额外配置
            stream_tokens (bool, optional): 在生成时流式传输token
                默认值: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: 来自代理的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/agents/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            # Don't yield empty string tokens as they cause generator issues
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        为运行创建反馈记录。

        这是LangSmith create_feedback API的简单包装器，因此
        凭证可以在服务中存储和管理，而不是在客户端中。
        参见: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str) -> ChatHistory:
        """
        获取聊天历史。

        Args:
            thread_id (str, optional): 用于标识对话的线程ID
        """
        request = ChatHistoryInput(thread_id=thread_id)
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    async def aupload_files_and_create_vector_db(
        self,
        files: List[tuple],  # List of (filename, file_content) tuples
        db_name: str = "chroma_db_uploaded",
        chunk_size: int = 2000,
        overlap: int = 500,
        use_local_embedding: bool = True,
        model_name: str = "BAAI/bge-m3",
        auto_switch: bool = True,
        db_type: str = "qdrant",
    ) -> dict[str, Any]:
        """
        上传文件并创建向量数据库（异步）
        
        Args:
            files: 文件列表，每个元素是 (filename, file_content) 元组
            db_name: 向量数据库名称
            chunk_size: 文本块大小
            overlap: 文本块重叠
            use_local_embeddings: 是否使用本地 embedding 模型
            model_name: 本地模型名称
        
        Returns:
            包含处理结果的字典
        """
        files_data = []
        for filename, content in files:
            files_data.append(("files", (filename, content, "application/octet-stream")))
        
        data = {
            "db_name": db_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "use_local_embedding": use_local_embedding,
            "model_name": model_name,
            "auto_switch": auto_switch,
            "db_type": db_type,
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 增加超时时间，文件处理可能较慢
            try:
                response = await client.post(
                    f"{self.base_url}/vector-db/upload",
                    files=files_data,
                    data=data,
                    headers=self._headers,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error uploading files: {e}")
        
        return response.json()

    def upload_files_and_create_vector_db(
        self,
        files: List[tuple],
        db_name: str = "chroma_db_uploaded",
        chunk_size: int = 2000,
        overlap: int = 500,
        use_local_embedding: bool = False,
        model_name: str = "BAAI/bge-small-en-v1.5",
        auto_switch: bool = True,
        db_type: str = "qdrant",
    ) -> dict[str, Any]:
        """
        上传文件并创建向量数据库（同步）
        """
        import asyncio
        return asyncio.run(
            self.aupload_files_and_create_vector_db(
                files=files,
                db_name=db_name,
                chunk_size=chunk_size,
                overlap=overlap,
                use_local_embedding=use_local_embedding,
                model_name=model_name,
                auto_switch=auto_switch,
                db_type=db_type,
            )
        )
    
    async def aswitch_vector_db(
        self,
        db_path: str,
        db_type: str = "qdrant",  # 新增参数
        collection_name: Optional[str] = None,  # 新增参数
    ) -> dict[str, Any]:
        """
        切换向量数据库路径（异步）
        
        Args:
            db_path: 新的数据库路径
            db_type: 数据库类型 ("chroma" 或 "qdrant")
            collection_name: 集合名（仅 Qdrant 需要）
        """
        data = {
            "db_path": db_path,
            "db_type": db_type,
        }
        if collection_name:
            data["collection_name"] = collection_name
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/vector-db/switch",
                    data=data,
                    headers=self._headers,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error switching vector database: {e}")

    def switch_vector_db(self, db_path: str) -> dict[str, Any]:
        """
        切换向量数据库路径（同步）
        """
        import asyncio
        return asyncio.run(self.aswitch_vector_db(db_path))




