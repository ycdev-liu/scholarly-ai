import inspect
import json
import logging
from re import T
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional
from uuid import UUID, uuid4

from click.core import V
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from sympy import im

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from core import settings
from core import setup_logging
import logging
from core.settings import LogLevel
import logging
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


logger=logging.getLogger(__name__)

logger=setup_logging(log_level=LogLevel.INFO, log_file="service.log", log_dir="./logs", enable_file_logging=True, enable_console_logging=False,root_logger=logger)

def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)



# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"kife checkpointer initialized successfully")
    logger.debug(f"lifespan started")
    logger.warning(f"lifespan started")
    logger.error(f"lifespan started")
    logger.critical(f"lifespan started")
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
                logger.info(f"Database checkpointer initialized successfully (type: {type(saver).__name__})")
            else:
                logger.info(f"Database checkpointer initialized (type: {type(saver).__name__}, no setup required)")
            
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()
                logger.info(f"Store initialized successfully (type: {type(store).__name__})")
            else:
                logger.info(f"Store initialized (type: {type(store).__name__}, no setup required)")

            # Configure agents with both memory components and async loading
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # Continue with other agents rather than failing startup

                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            
            # Test database connection after agents are configured
            try:
                test_config = RunnableConfig(configurable={"thread_id": "__init_test__"})
                # Try to get state (this will test the database connection)
                test_agent = get_agent(DEFAULT_AGENT)
                if hasattr(test_agent, "aget_state"):
                    test_state = await test_agent.aget_state(config=test_config)
                    logger.info("Database connection test successful - checkpointer is working")
                else:
                    logger.debug("Database connection test skipped (agent does not support state query)")
            except Exception as e:
                logger.warning(f"Database connection test failed (this may be normal on first run): {e}")
            logger.info("All agents configured with memory components successfully")
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)

router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    user_id = user_input.user_id or str(uuid4())
    

    configurable = {"thread_id": thread_id, "user_id": user_id}
    # 如果用户输入的模型不为空，则设置模型
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks = []
    # 如果开启了Langfuse追踪，则添加Langfuse回调
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    # 如果用户输入的agent_config不为空，则更新configurable
    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        # 如果用户输入的agent_config包含保留关键字，则抛出错误
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    # 创建RunnableConfig
    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    # 获取状态
    state = await agent.aget_state(config=config)
    # 获取中断任务
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]
    # 如果存在中断任务，则创建命令
    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """

    agent: AgentGraph = get_agent(agent_id)
    # 处理输入
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:

    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    logger=logging.getLogger(__name__)

    logger=setup_logging(log_level=LogLevel.INFO, log_file="service.log", log_dir="./logs", enable_file_logging=True, enable_console_logging=False,root_logger=logger)

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    # Check database status (checkpointer and store separately)
    try:
        from core.settings import DatabaseType
        from pathlib import Path
        
        db_info: dict[str, Any] = {}
        
        # Get actual database types being used
        checkpointer_type = settings.CHECKPOINTER_DB_TYPE or settings.DATABASE_TYPE
        store_type = settings.STORE_DB_TYPE or settings.DATABASE_TYPE
        
        # Checkpointer status (short-term memory)
        checkpointer_info: dict[str, Any] = {
            "type": checkpointer_type.value,
            "purpose": "short-term memory (conversation history)"
        }
        
        if checkpointer_type == DatabaseType.SQLITE:
            db_path = Path(settings.SQLITE_DB_PATH)
            checkpointer_info["file_exists"] = str(db_path.exists())
            if db_path.exists():
                checkpointer_info["file_size"] = f"{db_path.stat().st_size} bytes"
                checkpointer_info["file_path"] = str(db_path)
        
        # Test checkpointer connection
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
        
        # Store status (long-term memory)
        store_info: dict[str, Any] = {
            "type": store_type.value,
            "purpose": "long-term memory (cross-conversation knowledge)"
        }
        
        if store_type == DatabaseType.POSTGRES:
            store_info["host"] = str(settings.POSTGRES_HOST or "not_configured")
            store_info["database"] = str(settings.POSTGRES_DB or "not_configured")
            # Test PostgreSQL connection if configured
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

from fastapi import UploadFile,File,Form
from typing import List
from pathlib import Path
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,

)

@router.post("/vector-db/upload")
async def upload_files_to_vector_db(
    files: List[UploadFile]=File(...),
    db_name:str=Form("qdrant_db_uploader"),
    chunk_size:int = Form(2000),
    overlap:int =  Form(500),
    use_local_embedding : bool = Form(True),
    model_name:str = Form("BAAI/bge-m3"),
    auto_switch:bool = Form(False),
    db_type:str = Form("qdrant")
    ):
    result = {
        "success":False,
        "db_name":db_name,
        "total_files":len(files),
        "total_chunks":0,
        "processed_files":[],
        "errors":[]
    }

    try:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = []
            for upload_file in files :
                file_path = Path(temp_dir) / upload_file.filename
                with open(file_path,"wb") as f:
                    content = await upload_file.read()
                    f.write(content)
                saved_files.append(file_path)
 
            if use_local_embedding:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from core.settings import settings
                # 使用统一的配置路径
                cache_folder = settings.EMBEDDING_MODEL_CACHE_DIR
                # 如果是相对路径，转换为绝对路径（相对于项目根目录）
                if not os.path.isabs(cache_folder):
                    cache_folder = os.path.join(os.getcwd(), cache_folder.lstrip("./"))
                
                # 检查本地是否已有模型（优先使用本地模型）
                model_dir_name = f"models--{model_name.replace('/', '--')}"
                model_path = os.path.join(cache_folder, "hub", model_dir_name)
                local_model_exists = os.path.exists(model_path) and os.path.isdir(model_path)
                
                if local_model_exists:
                    # 优先使用本地已下载的模型（离线模式）
                    logger.info(f"检测到本地已下载的模型: {model_path}")
                    logger.info("优先使用本地模型（离线模式）...")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    try:
                        embeddings = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=cache_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                        logger.info(f"✅ 成功从本地加载模型: {model_name}")
                    except Exception as offline_error:
                        logger.warning(f"本地模型加载失败，尝试在线模式: {offline_error}")
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        embeddings = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=cache_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                else:
                    # 本地没有模型，尝试在线下载
                    logger.info(f"本地未找到模型，尝试从 HuggingFace 下载: {model_name}")
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        cache_folder=cache_folder,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
            else:
                pass
                # from langchain_openai import OpenAIEmbeddings 
                # embeddings = OpenAIEmbeddings()
            
            # 统一的向量数据库文件夹
            VECTOR_DB_BASE_DIR = "./vector_databases"
            
            # 确保统一文件夹存在
            os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
            
            db_path = Path(db_name)

            if not db_path.is_absolute():
                # 如果路径不是绝对路径，检查是否已经在统一文件夹下
                # 如果不在，则将其放在统一文件夹下
                if not str(db_path).startswith(VECTOR_DB_BASE_DIR):
                    db_path = Path(VECTOR_DB_BASE_DIR) / db_path.name
                else:
                    db_path = Path(os.getcwd()) / db_path
            # 如果是绝对路径，保持原样（向后兼容）
            
            db_type = db_type.lower()
            
            # 只支持 Qdrant
            if db_type != "qdrant":
                result['errors'].append(f"不支持的数据库类型: {db_type}。只支持: qdrant")
                return result

            # 如果数据库已存在，需要先清除缓存并关闭连接，然后才能安全删除
            if db_path.exists():
                try:
                    # 清除可能正在使用的数据库连接缓存
                    from agents.tools import clear_retriever_cache
                    clear_retriever_cache()
                    
                    # 等待一小段时间，确保文件句柄被释放
                    import time
                    import gc
                    gc.collect()  # 强制垃圾回收，帮助释放文件句柄
                    time.sleep(0.5)
                    
                    # 尝试删除，如果失败则重试（Windows 上文件锁定问题较常见）
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            # 在 Windows 上，先尝试重命名再删除，有时可以绕过文件锁定
                            if os.name == 'nt':  # Windows
                                try:
                                    # 尝试重命名为临时名称
                                    temp_name = str(db_path) + f".deleting_{int(time.time())}"
                                    if os.path.exists(temp_name):
                                        shutil.rmtree(temp_name, ignore_errors=True)
                                    os.rename(str(db_path), temp_name)
                                    shutil.rmtree(temp_name, ignore_errors=True)
                                    break
                                except (OSError, PermissionError):
                                    # 如果重命名失败，直接尝试删除
                                    shutil.rmtree(db_path)
                                    break
                            else:
                                shutil.rmtree(db_path)
                                break
                        except (OSError, PermissionError) as e:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 0.5  # 递增等待时间
                                logger.warning(f"删除数据库失败，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{max_retries}): {e}")
                                time.sleep(wait_time)
                                gc.collect()  # 再次尝试垃圾回收
                            else:
                                # 最后一次尝试失败，返回错误
                                error_msg = f"无法删除旧数据库（文件可能被其他程序占用）: {str(e)}"
                                logger.error(error_msg)
                                result['errors'].append(error_msg)
                                return result
                except Exception as e:
                    logger.error(f"删除旧数据库时出错: {e}")
                    result['errors'].append(f"无法删除旧数据库: {str(e)}")
                    return result

            # 创建 Qdrant 向量存储
            try:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance,VectorParams
            except ImportError:
                result['errors'].append("请安装langchain-qdrant和qdrant-client")
                return result

            client = QdrantClient(path=str(db_path))
            embedding_dim = len(embeddings.embed_query("test"))

            collection_name = "documents"
            try :
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embedding_dim,distance=Distance.COSINE),
                )
            
            vector_store = QdrantVectorStore(
                client= client,
                collection_name = collection_name,
                embedding = embeddings,
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = overlap
            )

            for file_path in saved_files:
                filename = file_path.name
                try:
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif filename.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    elif filename.endswith(".txt"):
                        try:
                            loader = TextLoader(file_path, encoding="utf-8")
                        except UnicodeDecodeError:
                            try:
                                loader = TextLoader(file_path, encoding="gbk")
                            except UnicodeDecodeError:
                                loader = TextLoader(file_path, encoding="latin-1")
                    else:
                        result["errors"].append(f"不支持数据类型: {filename}")
                        continue
                    
                    documents =loader.load()
                    chunks = text_splitter.split_documents(documents)

                    if chunks:
                        vector_store.add_documents(chunks)
                        result["total_chunks"] += len(chunks)
                        result["processed_files"].append(
                            {"filename":filename,"chunks":len(chunks)}
                        )
                except Exception as e:
                    result['errors'].append(f"Error loading file {filename}: {str(e)}")
                    continue


            result["success"] = True
            result["db_path"] = str(db_path)
            result['db_type'] = db_type


        
            if auto_switch == True:
                try:
                    from agents.tools import clear_retriever_cache
                    # 更新环境变量
                    os.environ["VECTOR_DB_TYPE"] = "qdrant"
                    os.environ["QDRANT_PATH"] = str(db_path)
                    os.environ["QDRANT_COLLECTION"] = collection_name
                    
                    # 清除缓存，强制重新加载
                    clear_retriever_cache()
                    
                    result["switched"] = True
                    result["message"] = f"数据库创建成功并已自动切换到: {db_path}"
                    logger.info(f"Database created and auto-switched to: {db_path} (type: {db_type})")
                except Exception as e:
                    result["switched"] = False
                    result["switch_error"] = str(e)
                    logger.warning(f"Database created but failed to auto-switch: {e}")
            else:
                result['auto_switch'] = False
                result['message'] = "向量数据库创建成功，但未自动切换"


    except Exception as e:
        result["errors"].append(f"创建向量数据库出错：{str(e)}")
        logger.error(f"error creating vector database:{e}",exc_info=True)
    return result

from agents.tools import clear_retriever_cache
import os
from typing import Optional

@router.post("/vector-db/switch")
async def switch_vector_db(
    db_type: str = Form(...),
    db_path: str = Form(...),
    collection_name: Optional[str] = Form(None)
):
    """
    切换向量数据库类型和路径
    
    Args:
        db_type: 数据库类型（只支持 "qdrant"）
        db_path: 数据库路径
        collection_name: 集合名（默认 "documents")
    
    Returns:
        切换结果
    """
    from agents.tools import clear_retriever_cache

    try:
        db_type = db_type.lower()
        if db_type != "qdrant":
            raise HTTPException(status_code=400, detail="Invalid database type. Only supports: qdrant")
        
        # 设置 VECTOR_DB_TYPE 环境变量
        os.environ["VECTOR_DB_TYPE"] = "qdrant"
        os.environ["QDRANT_PATH"] = db_path
        os.environ["QDRANT_COLLECTION"] = collection_name or "documents"
        logger.info(f"Switched to Qdrant database: {db_path} with collection: {collection_name or 'documents'}")
        
        # 清除缓存，强制重新加载
        clear_retriever_cache()
        
        return {
            "success": True,
            "message": f"Vector database switched to: {db_path} (type: qdrant)",
            "db_path": db_path,
            "db_type": "qdrant",
            "collection_name": collection_name or "documents"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching vector database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error switching vector database: {str(e)}")

app.include_router(router)
