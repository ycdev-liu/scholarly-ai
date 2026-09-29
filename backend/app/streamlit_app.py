import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import streamlit as st
from client import AgentClient, AgentClientError
from dotenv import load_dotenv
from pydantic import ValidationError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Scholarly AI"
APP_ICON = "📚"
USER_ID_COOKIE = "user_id"

AGENT_LABELS = {
    "auto": "智能研究",
    "literature-review": "文献综述",
    "openreview-agent": "论文检索",
    "rag-assistant": "本地论文问答",
    "paper-research-supervisor": "完整研究流程",
}

AGENT_DESCRIPTIONS = {
    "auto": "优先检索本地论文，证据不足时自动扩展到外部学术来源。",
    "literature-review": "检索多篇论文、核对证据并撰写带来源链接的综述。",
    "openreview-agent": "面向 OpenReview 和 arXiv 检索、筛选与下载论文。",
    "rag-assistant": "基于已下载和已建库的论文进行定向问答。",
    "paper-research-supervisor": "编排论文搜索、下载、建库和内容研读。",
}

QUICK_PROMPTS = {
    "auto": [
        (
            ":material/trending_up: 追踪研究前沿",
            "检索近三年大语言模型检索增强生成的代表性论文，并按技术路线归类。",
        ),
        (
            ":material/compare_arrows: 对比技术路线",
            "对比当前文献中常见的 RAG 重排方法，总结各自优势、局限和适用场景。",
        ),
        (
            ":material/menu_book: 研读经典论文",
            "帮我查找并研读 Attention Is All You Need，重点解释其核心创新。",
        ),
        (
            ":material/inventory_2: 盘点本地文献",
            "检查当前已下载的论文和本地知识库，告诉我现在可以研读哪些内容。",
        ),
    ],
    "openreview-agent": [
        (
            ":material/search: 按主题找论文",
            "搜索关于多模态大模型的高相关论文，列出标题、作者、会议和摘要要点。",
        ),
        (
            ":material/event: 按会议筛选",
            "查找最近一届 ICLR 中与 AI Agent 相关的论文，按相关性排序。",
        ),
        (
            ":material/download: 下载 arXiv 论文",
            "下载 arXiv:1706.03762 到本地论文目录。",
        ),
        (
            ":material/summarize: 整理候选清单",
            "检索小样本学习的代表性论文，精选 5 篇并说明推荐理由。",
        ),
    ],
    "rag-assistant": [
        (
            ":material/description: 总结本地论文",
            "根据当前本地论文库，总结核心研究问题、方法和主要结论。",
        ),
        (
            ":material/account_tree: 梳理方法框架",
            "从当前论文库中梳理各篇论文的方法框架，并说明它们的关系。",
        ),
        (
            ":material/table_view: 提取实验结果",
            "提取当前论文中的数据集、评价指标、基线和主要实验结果。",
        ),
        (
            ":material/help: 检查可用资料",
            "列出已下载的论文和当前向量数据库状态。",
        ),
    ],
    "paper-research-supervisor": [
        (
            ":material/travel_explore: 开展主题调研",
            "围绕长文本上下文建模开展一次小型调研：搜索、筛选代表性论文，并归纳主要路线。",
        ),
        (
            ":material/download_for_offline: 搜索并入库",
            "搜索神经符号学习的代表性论文，下载最相关的论文并建立本地向量库。",
        ),
        (
            ":material/rate_review: 形成文献综述",
            "基于当前可用论文，形成一份包含问题背景、方法分类和研究空白的综述提纲。",
        ),
        (
            ":material/fact_check: 核对研究结论",
            "根据本地论文内容核对主要结论，并标明支持每个结论的论文。",
        ),
    ],
}

TOOL_LABELS = {
    "OpenReview_Search": "检索学术论文",
    "Download_Paper": "下载 OpenReview 论文",
    "Download_Paper_From_ArXiv": "下载 arXiv 论文",
    "List_Downloaded_Papers": "盘点本地论文",
    "Create_Vector_DB_From_PDF": "建立论文索引",
    "Database_Search": "检索本地论文",
    "Get_Vector_DB_Info": "读取知识库状态",
    "Switch_Vector_DB": "切换知识库",
    "transfer_to_openreview_agent": "论文检索专员",
    "transfer_to_rag_assistant": "本地论文助手",
}

MODEL_LABELS = {
    "fake": "演示模型（fake）",
}


def apply_styles() -> None:
    st.html(
        """
        <style>
        :root {
            --scholar-ink: #17211b;
            --scholar-green: #1f5c43;
            --scholar-green-dark: #153e2e;
            --scholar-mint: #e8f2ec;
            --scholar-coral: #c95945;
            --scholar-paper: #f7faf8;
            --scholar-line: #d8e2dc;
            --scholar-muted: #5f6f66;
        }

        html, body {
            font-family: "Noto Sans SC", "Microsoft YaHei", "PingFang SC", sans-serif;
            letter-spacing: 0;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--scholar-paper);
            color: var(--scholar-ink);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            background: var(--scholar-green-dark);
            border-right: 1px solid #2f5947;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #f4f8f5;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #b9cdc1;
        }

        [data-testid="stSidebar"] hr {
            border-color: #37614f;
        }

        .stButton > button {
            min-height: 2.75rem;
            border-radius: 6px;
            border-color: var(--scholar-line);
            font-weight: 600;
            letter-spacing: 0;
        }

        .stButton > button:hover {
            border-color: var(--scholar-green);
            color: var(--scholar-green);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #f2f7f4;
            border-color: #f2f7f4;
            color: var(--scholar-green-dark);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            border-color: #668676;
            color: #f4f8f5;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #ffffff;
        }

        [data-testid="stSelectbox"] > div > div,
        [data-testid="stChatInput"] textarea {
            border-radius: 6px;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--scholar-line);
            background: rgba(247, 250, 248, 0.96);
        }

        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid var(--scholar-line);
            border-radius: 8px;
            margin-bottom: 0.75rem;
            padding: 0.75rem;
        }

        [data-testid="stChatMessageContent"] {
            min-width: 0;
            overflow-wrap: anywhere;
        }

        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        .scholar-brand {
            padding: 0.25rem 0 0.75rem;
        }

        .scholar-brand__name {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.35;
        }

        .scholar-brand__tagline {
            color: #b9cdc1;
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }

        .research-header {
            border-bottom: 1px solid var(--scholar-line);
            margin-bottom: 1.5rem;
            padding-bottom: 1.25rem;
        }

        .research-header__meta {
            align-items: center;
            color: var(--scholar-green);
            display: flex;
            font-size: 0.78rem;
            font-weight: 700;
            gap: 0.5rem;
            margin-bottom: 0.55rem;
        }

        .research-header__dot {
            background: var(--scholar-coral);
            border-radius: 50%;
            display: inline-block;
            height: 0.5rem;
            width: 0.5rem;
        }

        .research-header h1 {
            color: var(--scholar-ink);
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.2;
            margin: 0;
        }

        .research-header p {
            color: var(--scholar-muted);
            font-size: 1rem;
            line-height: 1.7;
            margin: 0.65rem 0 0;
            max-width: 720px;
        }

        .quick-heading {
            color: var(--scholar-ink);
            font-size: 0.95rem;
            font-weight: 700;
            margin: 1.9rem 0 0.75rem;
        }

        .service-state {
            align-items: center;
            color: #b9cdc1;
            display: flex;
            font-size: 0.78rem;
            gap: 0.45rem;
            margin-top: 1.25rem;
        }

        .service-state__dot {
            background: #68c18c;
            border-radius: 50%;
            display: inline-block;
            height: 0.45rem;
            width: 0.45rem;
        }

        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.25rem;
            }

            .research-header h1 {
                font-size: 1.75rem;
            }

            .stButton > button {
                min-height: 3rem;
                white-space: normal;
            }
        }
        </style>
        """
    )


def format_tool_name(tool_name: str) -> str:
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))


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
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={},
    )
    apply_styles()
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
            host = os.getenv("HOST", "localhost")
            # 0.0.0.0 只能用于服务端监听，客户端应连 localhost
            if host in ("0.0.0.0", "::"):
                host = "127.0.0.1"
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("正在连接研究服务…"):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"无法连接研究服务 {agent_url}：{e}")
            st.caption("请确认后端服务已启动，然后刷新页面。")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("未找到该研究会话的历史记录。")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Research workspace controls
    with st.sidebar:
        st.markdown(
            """
            <div class="scholar-brand">
                <div class="scholar-brand__name">Scholarly AI</div>
                <div class="scholar-brand__tagline">学术研究工作台</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            ":material/add: 新建研究",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.session_state.pending_approval = None
            st.session_state.thread_id = str(uuid.uuid4())
            st.query_params.pop("thread_id", None)
            st.rerun()

        st.markdown("#### 研究模式")
        agent_list = [a.key for a in agent_client.info.agents]
        selected_agent = agent_client.agent or agent_client.info.default_agent
        agent_idx = agent_list.index(selected_agent) if selected_agent in agent_list else 0
        agent_client.agent = st.selectbox(
            "研究模式",
            options=agent_list,
            index=agent_idx,
            format_func=lambda key: AGENT_LABELS.get(key, key),
            label_visibility="collapsed",
        )
        st.caption(
            AGENT_DESCRIPTIONS.get(
                agent_client.agent,
                next(
                    (
                        agent.description
                        for agent in agent_client.info.agents
                        if agent.key == agent_client.agent
                    ),
                    "",
                ),
            )
        )

        st.divider()
        st.markdown("#### 会话设置")
        model_idx = (
            agent_client.info.models.index(agent_client.info.default_model)
            if agent_client.info.default_model in agent_client.info.models
            else 0
        )
        model = st.selectbox(
            "推理模型",
            options=agent_client.info.models,
            index=model_idx,
            format_func=lambda name: MODEL_LABELS.get(name, name),
        )
        use_streaming = st.toggle(
            "流式输出（实验）",
            value=False,
            help="若流式连接被页面刷新中断，请关闭此选项。",
        )

        @st.dialog("分享研究会话")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.caption("使用以下链接继续当前研究会话。")
            st.code(chat_url, language=None)

        if st.button(":material/share: 分享当前会话", use_container_width=True):
            share_chat_dialog()

        with st.expander("会话信息"):
            st.caption("会话 ID")
            st.code(st.session_state.thread_id, language=None)
            st.caption("用户 ID")
            st.code(user_id, language=None)

        st.markdown(
            """
            <div class="service-state">
                <span class="service-state__dot"></span>
                研究服务已连接
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    agent_label = AGENT_LABELS.get(agent_client.agent, agent_client.agent)
    st.markdown(
        f"""
        <section class="research-header">
            <div class="research-header__meta">
                <span class="research-header__dot"></span>
                {agent_label}
            </div>
            <h1>学术研究工作台</h1>
            <p>{AGENT_DESCRIPTIONS.get(agent_client.agent, "从研究问题出发，组织论文和证据。")}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    preview_root = Path("data/vector_databases")
    preview_stores = (
        sorted(path for path in preview_root.iterdir() if (path / "previews").is_dir())
        if preview_root.is_dir()
        else []
    )
    if preview_stores:
        @st.dialog("论文图页", width="large")
        def show_paper_previews() -> None:
            store = st.selectbox("向量库", preview_stores, format_func=lambda path: path.name)
            images = sorted((store / "previews").glob("page-*.png"))
            if images:
                image = st.selectbox(
                    "页面",
                    images,
                    format_func=lambda path: f"第 {int(path.stem.removeprefix('page-'))} 页",
                )
                st.image(str(image), use_container_width=True)
                st.download_button(
                    "下载图页",
                    data=image.read_bytes(),
                    file_name=image.name,
                    mime="image/png",
                    icon=":material/download:",
                )

        if st.button("查看论文图页", icon=":material/image:"):
            show_paper_previews()

    if set(agent_client.info.models) == {"fake"}:
        st.warning(
            "当前为演示模式：`fake` 只用于验证请求链路，"
            "不会生成真实的学术回答。",
            icon=":material/info:",
        )

    suggested_prompt = None
    quick_start = st.empty()
    if len(messages) == 0:
        with quick_start.container():
            st.markdown('<div class="quick-heading">快速开始</div>', unsafe_allow_html=True)
            quick_prompts = QUICK_PROMPTS.get(agent_client.agent, QUICK_PROMPTS["auto"])
            for row_start in range(0, len(quick_prompts), 2):
                columns = st.columns(2)
                for column, (label, prompt) in zip(columns, quick_prompts[row_start : row_start + 2]):
                    with column:
                        if st.button(
                            label,
                            key=f"quick-prompt-{agent_client.agent}-{row_start}-{label}",
                            help=prompt,
                            use_container_width=True,
                        ):
                            suggested_prompt = prompt

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if agent_client.agent == "literature-review":
        try:
            st.session_state.pending_approval = await agent_client.aget_pending_approval(
                st.session_state.thread_id
            )
        except AgentClientError:
            pass

    pending = st.session_state.get("pending_approval")
    if pending and agent_client.agent == "literature-review":
        action = pending.get("action", {})
        st.warning(f"待确认：{action.get('kind', '操作')} {action.get('target', '')}")
        approve_col, deny_col = st.columns(2)
        decision = None
        if approve_col.button("批准", icon=":material/check:", use_container_width=True):
            decision = "approve"
        if deny_col.button("拒绝", icon=":material/close:", use_container_width=True):
            decision = "deny"
        if decision:
            try:
                response = await agent_client.ainvoke(
                    message="", approval=decision, model=model,
                    thread_id=st.session_state.thread_id, user_id=user_id,
                )
                st.session_state.pending_approval = None
                messages.append(response)
                st.rerun()
            except AgentClientError as e:
                st.error(f"操作确认失败：{e}")
        return

    # Generate new message if the user provided new input
    prompt_placeholder = {
        "auto": "输入一个研究问题…",
        "literature-review": "描述你要综述的研究问题…",
        "openreview-agent": "输入主题、会议、作者或 arXiv ID…",
        "rag-assistant": "询问本地论文中的内容…",
        "paper-research-supervisor": "描述你想完成的研究任务…",
    }.get(agent_client.agent, "输入研究问题…")
    user_input = suggested_prompt or st.chat_input(prompt_placeholder)
    if user_input:
        quick_start.empty()
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            with st.spinner("正在检索论文并组织回答…"):
                if use_streaming:
                    stream = agent_client.astream(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        user_id=user_id,
                    )
                    try:
                        await draw_messages(stream, is_new=True)
                    finally:
                        await stream.aclose()
                    st.rerun()
                else:
                    response = await agent_client.ainvoke(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        user_id=user_id,
                    )
                    messages.append(response)
                    if response.type == "custom" and response.custom_data.get("kind") == "approval_required":
                        st.session_state.pending_approval = response.custom_data
                        st.rerun()
                    else:
                        st.rerun()
        except AgentClientError as e:
            st.error(f"研究请求执行失败：{e}")
            st.stop()
        except RuntimeError as e:
            st.error("流式输出已中断，请关闭“流式输出（实验）”后重试。")
            st.caption(str(e))
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
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"收到了无法识别的消息：{type(msg)}")
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
                if msg.content:
                    st.session_state.pending_approval = None
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

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
                            tool_name = format_tool_name(tool_call["name"])
                            if "transfer_to" in tool_call["name"]:
                                label = f"研究分工 · {tool_name}"
                            else:
                                label = f"正在执行 · {tool_name}"

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
                            status.write("**输入**")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"收到了意外的消息类型：{tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("**结果**")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                if msg.custom_data.get("kind") == "approval_required":
                    if is_new:
                        st.session_state.messages.append(msg)
                    st.session_state.pending_approval = msg.custom_data
                    st.info(
                        f"等待确认：{msg.custom_data['action']['kind']} "
                        f"{msg.custom_data['action']['target']}"
                    )
                    continue
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("收到了无法识别的任务数据。")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message("task", avatar="assistant")
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"收到了意外的消息类型：{msg.type}")
                st.write(msg)
                st.stop()

    if is_new and streaming_content:
        st.session_state.messages.append(ChatMessage(type="ai", content=streaming_content))


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    if not (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")):
        return

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    if not latest_run_id:
        return
    feedback = st.segmented_control(
        "回答评分", options=[1, 2, 3, 4, 5], key=f"feedback-{latest_run_id}"
    )

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        normalized_score = feedback / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"反馈提交失败：{e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("反馈已记录", icon=":material/reviews:")


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
            popover.write("**结果**")
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
                            f"研究分工 · {format_tool_name(tc['name'])}",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(
                            format_tool_name(tc["name"]), icon=":material/build:"
                        )
                        popover.write(f"**工具：** {format_tool_name(tc['name'])}")
                        popover.write("**输入**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
