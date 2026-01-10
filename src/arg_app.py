import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

import tempfile
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"

# 统一的向量数据库文件夹
VECTOR_DB_BASE_DIR = "./vector_databases"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        
        # Retry connection with exponential backoff
        max_retries = 5
        retry_delay = 2
        connected = False
        
        with st.spinner("Connecting to agent service..."):
            for attempt in range(max_retries):
                try:
                    st.session_state.agent_client = AgentClient(base_url=agent_url)
                    connected = True
                    break
                except AgentClientError as e:
                    if attempt < max_retries - 1:
                        # Wait before retrying (exponential backoff)
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # Last attempt failed
                        st.error(f"Error connecting to agent service at {agent_url}: {e}")
                        st.markdown(
                            f"""
                            **Connection Failed After {max_retries} Attempts**
                            
                            The agent service might still be starting up. Please:
                            1. Wait a few more seconds
                            2. Check if the service is running: `docker compose ps`
                            3. Check service logs: `docker compose logs agent_service`
                            4. Refresh this page to retry
                            """
                        )
                        st.stop()
        
        if not connected:
            st.error("Failed to connect to agent service")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages_list: list[ChatMessage] = []
        else:
            try:
                history = agent_client.get_history(thread_id=thread_id)
                messages_list = history.messages if history else []
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages_list = []
        st.session_state.messages = messages_list
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            if agent_client.info and agent_client.info.models:
                model_idx = agent_client.info.models.index(agent_client.info.default_model) if agent_client.info.default_model in agent_client.info.models else 0
                model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            else:
                model = None
                st.warning("无法获取模型列表")
            
            if agent_client.info and agent_client.info.agents:
                agent_list = [a.key for a in agent_client.info.agents]
                agent_idx = agent_list.index(agent_client.info.default_agent) if agent_client.info.default_agent in agent_list else 0
                agent_client.agent = st.selectbox(
                    "Agent to use",
                    options=agent_list,
                    index=agent_idx,
                )
            else:
                st.warning("无法获取 Agent 列表")
            
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            try:
                session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]  # type: ignore
                st_base_url = urllib.parse.urlunparse(
                    [session.client.request.protocol, session.client.request.host, "", "", "", ""]  # type: ignore
                )
            except Exception:
                st_base_url = "http://localhost:8501"
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

        # ========== 文件上传和向量数据库管理 ==========
        with st.expander(":material/upload_file: 上传文件并创建向量数据库", expanded=False):
            st.markdown("### 📁 上传文档")
            st.markdown("支持格式: PDF, DOCX, TXT")
            
            uploaded_files = st.file_uploader(
                "选择文件",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="可以一次上传多个文件"
            )
            
            if uploaded_files:
                st.info(f"已选择 {len(uploaded_files)} 个文件")
                for file in uploaded_files:
                    st.text(f"  • {file.name} ({file.size / 1024:.1f} KB)")
            
            # 数据库配置选项
            st.markdown("### ⚙️ 数据库配置")
            
            # 新增：数据库类型选择
            db_type = st.selectbox(
                "数据库类型",
                options=["qdrant"],
                index=0,  # 默认 Qdrant
                help="选择要创建的向量数据库类型"
            )
            
            db_name = st.text_input(
                "数据库名称",
                value="",  # 留空则自动生成
                help="数据库名称（将存储在 vector_databases 文件夹下，留空则自动生成）"
            )
            
            chunk_size = st.slider(
                "文本块大小",
                min_value=500,
                max_value=5000,
                value=2000,
                step=500,
                help="每个文本块的最大字符数"
            )
            
            overlap = st.slider(
                "文本块重叠",
                min_value=0,
                max_value=1000,
                value=500,
                step=100,
                help="相邻文本块之间的重叠字符数"
            )
            
            use_local_embedding = st.toggle(  # 修复：改为单数
                "使用本地 Embedding 模型",
                value=True,  # 修复：改为 True，匹配后端默认值
                help="如果启用，使用本地模型（需要模型已下载到缓存）"
            )
            
            # 新增：模型名称输入
            model_name = st.text_input(
                "模型名称",
                value="BAAI/bge-m3",
                help="本地 embedding 模型名称"
            )
            
            # 新增：自动切换选项
            auto_switch = st.toggle(
                "创建后自动切换",
                value=True,
                help="创建数据库后自动切换到该数据库"
            )
            
            # 创建数据库按钮
            if st.button("🚀 创建向量数据库", use_container_width=True, type="primary"):
                if not uploaded_files:
                    st.error("请先上传文件！")
                else:
                    await create_vector_db_from_files(
                        uploaded_files=uploaded_files,
                        db_name=db_name,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        use_local_embedding=use_local_embedding,  # 修复：改为单数
                        model_name=model_name,  # 新增
                        auto_switch=auto_switch,  # 新增
                        db_type=db_type,  # 新增
                    )
        
        # 数据库选择器
        st.markdown("---")
        with st.popover(":material/storage: 向量数据库管理", use_container_width=True):
            # 显示当前使用的数据库
            current_db_path = st.session_state.get("current_db_path", 
                os.getenv("QDRANT_PATH", os.path.join(VECTOR_DB_BASE_DIR, "qdrant_db")))
            current_db_type = st.session_state.get("current_db_type", 
                os.getenv("VECTOR_DB_TYPE", "qdrant").lower())
            
            if current_db_path and os.path.exists(current_db_path):
                db_type_icon = "🔷" if current_db_type == "qdrant" else "🔶"
                st.info(f"{db_type_icon} **当前使用: {current_db_type.upper()}** 数据库\n`{current_db_path}`")
            
            st.markdown("---")
            
            # 获取数据库信息列表
            db_info_list = _get_available_databases_info()
            
            if db_info_list:
                # 创建带类型标签的选项列表
                db_options = []
                for info in db_info_list:
                    db_type_icon = "🔷" if info["type"] == "qdrant" else "🔶"
                    label = f"{db_type_icon} [{info['type'].upper()}] {info['path']}"
                    db_options.append(label)
                
                # 找到当前数据库的索引
                default_index = 0
                for idx, info in enumerate(db_info_list):
                    if info["path"] == current_db_path:
                        default_index = idx
                        break
                
                selected_label = st.selectbox(
                    "选择向量数据库",
                    options=db_options,
                    index=default_index,
                    help="选择要使用的向量数据库（🔷 QDRANT）"
                )
                
                # 获取选中的数据库信息
                selected_index = db_options.index(selected_label)
                selected_info = db_info_list[selected_index]
                selected_db = selected_info["path"]
                selected_db_type = selected_info["type"]
                
                # 显示选中数据库的详细信息
                with st.expander("📋 数据库详情", expanded=False):
                    st.markdown(f"**类型:** {selected_db_type.upper()}")
                    st.markdown(f"**路径:** `{selected_db}`")
                    if selected_db_type == "qdrant":
                        st.markdown(f"**集合名:** documents")
                
                if st.button("✅ 切换到该数据库", use_container_width=True, type="primary"):
                    with st.spinner(f"正在切换到 {selected_db_type.upper()} 数据库..."):
                        success = await switch_vector_database(
                            db_path=selected_db,
                            db_type=selected_db_type,
                            collection_name="documents" if selected_db_type == "qdrant" else None
                        )
                    
                    if success:
                        st.session_state["current_db_path"] = selected_db
                        st.session_state["current_db_type"] = selected_db_type
                        st.success(f"✅ 已切换到 **{selected_db_type.upper()}** 数据库！")
                        st.info(f"路径: `{selected_db}`\n\n💡 提示：新的查询将使用此数据库进行检索")
                        st.rerun()
                    else:
                        st.error("❌ 切换数据库失败，请检查后端服务状态或重试")
            else:
                st.info("暂无可用的向量数据库")
                st.markdown("""
                **提示：**
                - 使用"上传文件并创建向量数据库"功能可以创建新数据库
                - 或者确保数据库文件存在于项目目录中
                - 支持的数据库类型：Qdrant
                """)

    # Draw existing messages
    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                if st.session_state.last_message:
                    with st.session_state.last_message:
                        streaming_placeholder = st.empty()
                else:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                if st.session_state.last_message:
                    with st.session_state.last_message:
                        # If the message has content, write it out.
                        # Reset the streaming variables to prepare for the next message.
                        if msg.content:
                            if streaming_placeholder:
                                streaming_placeholder.write(msg.content)
                                streaming_content = ""
                                streaming_placeholder = None
                            else:
                                st.write(msg.content)

                        if msg.tool_calls:
                            # Create a status container for each tool call and store the
                            # status container by ID to ensure results are mapped to the
                            # correct status container.
                            call_results = {}
                            for tool_call in msg.tool_calls:
                                # Use different labels for transfer vs regular tool calls
                                if "transfer_to" in tool_call["name"]:
                                    label = f"""💼 Sub Agent: {tool_call["name"]}"""
                                else:
                                    label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                                status = st.status(
                                    label,
                                    state="running" if is_new else "complete",
                                )
                                call_results[tool_call["id"]] = status

                            # Expect one ToolMessage for each tool call.
                            for tool_call in msg.tool_calls:
                                if "transfer_to" in tool_call["name"]:
                                    status = call_results[tool_call["id"]]
                                    status.update(expanded=True)
                                    await handle_sub_agent_msgs(messages_agen, status, is_new)
                                    break

                                # Only non-transfer tool calls reach this point
                                status = call_results[tool_call["id"]]
                                status.write("Input:")
                                status.write(tool_call["args"])
                                tool_result_raw = await anext(messages_agen)
                                
                                if isinstance(tool_result_raw, str):
                                    st.error(f"Unexpected string message: {tool_result_raw}")
                                    continue
                                
                                tool_result: ChatMessage = tool_result_raw

                                if tool_result.type != "tool":
                                    st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                    st.write(tool_result)
                                    st.stop()

                                # Record the message if it's new, and update the correct
                                # status container with the result
                                if is_new:
                                    st.session_state.messages.append(tool_result)
                                if tool_result.tool_call_id:
                                    status = call_results[tool_result.tool_call_id]
                                status.write("Output:")
                                status.write(tool_result.content)
                                status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    # Check if there are messages and if the last message has a run_id
    if not st.session_state.messages:
        return
    
    latest_message = st.session_state.messages[-1]
    latest_run_id = latest_message.run_id if hasattr(latest_message, 'run_id') else None
    
    # Only show feedback widget if run_id is available
    if not latest_run_id:
        return
    
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


async def create_vector_db_from_files(
    uploaded_files: list,
    db_name: str = None,  # 如果为 None，将自动生成名称
    chunk_size: int = 2000,
    overlap: int = 500,
    use_local_embedding: bool = True,  # 修复：改为单数
    model_name: str = "BAAI/bge-m3",  # 新增
    auto_switch: bool = True,  # 新增
    db_type: str = "qdrant",  # 新增
) -> None:
    """
    从上传的文件创建向量数据库（通过后端 API）
    
    Args:
        uploaded_files: Streamlit 上传的文件列表
        db_name: 数据库名称（不包含路径，将自动放在 VECTOR_DB_BASE_DIR 下）
        chunk_size: 文本块大小
        overlap: 文本块重叠
        use_local_embedding: 是否使用本地 embedding 模型（注意：单数形式）
        model_name: 模型名称
        auto_switch: 是否自动切换到新创建的数据库
        db_type: 数据库类型（只支持 "qdrant"）
    """
    # 确保统一文件夹存在
    os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
    
    # 如果没有提供数据库名称，自动生成
    if db_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = f"{db_type}_db_{timestamp}"
    
    # 构建完整路径（统一放在 vector_databases 文件夹下）
    if not os.path.isabs(db_name):
        db_path = os.path.join(VECTOR_DB_BASE_DIR, db_name)
    else:
        db_path = db_name
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 准备文件数据
        status_text.text("📤 准备上传文件...")
        progress_bar.progress(10)
        
        agent_client: AgentClient = st.session_state.agent_client
        files_data = []
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getvalue()
            files_data.append((uploaded_file.name, file_content))
        
        status_text.text("📥 上传文件到服务器...")
        progress_bar.progress(20)
        
        # 调用后端 API（修复：使用正确的参数名和新增的参数，传递完整路径）
        result = await agent_client.aupload_files_and_create_vector_db(
            files=files_data,
            db_name=db_path,  # 使用完整路径
            chunk_size=chunk_size,
            overlap=overlap,
            use_local_embedding=use_local_embedding,  # 修复：改为单数
            model_name=model_name,  # 新增
            auto_switch=auto_switch,  # 新增
            db_type=db_type,  # 新增
        )
        
        progress_bar.progress(100)
        
        if result.get("success"):
            status_text.text("✅ 向量数据库创建完成！")
            
            # 显示数据库类型信息
            created_db_type = result.get("db_type", db_type)
            
            st.success(f"""
            **向量数据库创建成功！**
            
            - 📁 数据库路径: `{result.get('db_path', db_path)}`
            - 🗄️ 数据库类型: {created_db_type.upper()}
            - 📄 处理文件数: {result.get('total_files', 0)}
            - 📝 总文本块数: {result.get('total_chunks', 0)}
            - 🔧 Embedding 模型: {'本地模型' if use_local_embedding else 'OpenAI'} ({model_name})
            """)
            
            # 显示自动切换信息
            if result.get("switched"):
                st.info("✅ 已自动切换到新创建的数据库！")
            elif auto_switch:
                st.warning(f"⚠️ 自动切换失败: {result.get('switch_error', '未知错误')}")
            
            # 显示处理的文件
            if result.get("processed_files"):
                st.info("处理的文件：")
                for file_info in result["processed_files"]:
                    st.text(f"  ✅ {file_info['filename']}: {file_info['chunks']} 个文本块")
            
            # 显示错误（如果有）
            if result.get("errors"):
                st.warning("部分错误：")
                for error in result["errors"]:
                    st.text(f"  ⚠️ {error}")
            
            # 更新 session state
            if result.get("switched"):
                st.session_state["current_db_path"] = result.get("db_path", db_path)
                st.session_state["current_db_type"] = created_db_type
        else:
            status_text.text("❌ 创建向量数据库失败")
            st.error(f"创建向量数据库时出错: {', '.join(result.get('errors', ['未知错误']))}")
            
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ 发生错误")
        st.error(f"❌ 创建向量数据库时出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def get_available_databases() -> list[dict[str, str]]:
    """
    获取可用的向量数据库列表（返回详细信息）
    优先从统一的 vector_databases 文件夹查找，也兼容旧路径
    
    Returns:
        包含数据库路径和类型的字典列表
    """
    databases = []
    
    # 优先检查统一的向量数据库文件夹
    if os.path.exists(VECTOR_DB_BASE_DIR) and os.path.isdir(VECTOR_DB_BASE_DIR):
        for item in os.listdir(VECTOR_DB_BASE_DIR):
            db_path = os.path.join(VECTOR_DB_BASE_DIR, item)
            if os.path.isdir(db_path):
                # 检测数据库类型
                if os.path.exists(os.path.join(db_path, "config.json")):
                    databases.append({"path": db_path, "type": "qdrant"})

    
    # 兼容旧路径（向后兼容）
    legacy_paths = [
        "./qdrant_db",
    ]
    
    for db_path in legacy_paths:
        if os.path.exists(db_path) and os.path.isdir(db_path):
            # 检测数据库类型
            if os.path.exists(os.path.join(db_path, "config.json")):
                db_type = "qdrant"
                if not any(d["path"] == db_path for d in databases):
                    databases.append({"path": db_path, "type": db_type})
    
    return sorted(databases, key=lambda x: x["path"]) if databases else []

def _get_available_databases_info() -> list[dict[str, str]]:
    """
    获取可用的向量数据库详细信息
    优先从统一的 vector_databases 文件夹查找，也兼容旧路径
    
    Returns:
        包含数据库路径和类型的字典列表
    """
    databases = []
    
    # 优先检查统一的向量数据库文件夹
    if os.path.exists(VECTOR_DB_BASE_DIR) and os.path.isdir(VECTOR_DB_BASE_DIR):
        for item in os.listdir(VECTOR_DB_BASE_DIR):
            db_path = os.path.join(VECTOR_DB_BASE_DIR, item)
            if os.path.isdir(db_path):
                # 检测数据库类型
                if os.path.exists(os.path.join(db_path, "config.json")):
                    databases.append({"path": db_path, "type": "qdrant"})

    
    # 兼容旧路径（向后兼容）
    legacy_paths = [
        "./qdrant_db",
    ]
    
    for db_path in legacy_paths:
        if os.path.exists(db_path) and os.path.isdir(db_path):
            # 检测数据库类型
            if os.path.exists(os.path.join(db_path, "config.json")):
                if not any(d["path"] == db_path for d in databases):
                    databases.append({"path": db_path, "type": "qdrant"})
    
    return sorted(databases, key=lambda x: x["path"]) if databases else []

import httpx
import logging
logger = logging.getLogger(__name__)
async def switch_vector_database(
    db_path: str,
    db_type: str = None,  # 新增：数据库类型参数
    collection_name: str = None,  # 新增：集合名参数
) -> bool:
    """
    通过后端 API 切换向量数据库
    
    Args:
        db_path: 数据库路径
        db_type: 数据库类型（如果不提供，从路径推断）
        collection_name: 集合名（仅 Qdrant 需要）
    
    Returns:
        是否切换成功
    """
    try:
        agent_client: AgentClient = st.session_state.agent_client
        
        # 如果没有提供 db_type，从路径推断
        if db_type is None:
            if "qdrant" in db_path.lower():
                db_type = "qdrant"
            else:
                # 尝试检查目录内容判断类型
                if os.path.exists(os.path.join(db_path, "config.json")):
                    db_type = "qdrant"
                else:
                    # 默认使用 Qdrant
                    db_type = "qdrant"
        
        # 调用后端 API 切换数据库（修复：传递 db_type 和 collection_name）
        result = await agent_client.aswitch_vector_db(
            db_path=db_path,
            db_type=db_type,
            collection_name=collection_name or ("documents" if db_type == "qdrant" else None)
        )
        
        return result.get("success", False)
    except Exception as e:
        logger.error(f"Error switching database: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())
