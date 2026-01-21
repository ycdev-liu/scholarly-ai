from datetime import datetime
from typing import Literal, Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from agents.openreview_agent import openreview_agent
from agents.rag_assistant import rag_assistant
from core import get_model, settings


from dataclasses import dataclass, field

@dataclass
class ResearchContext:
    """存储研究任务的相关上下文信息"""
    # 下载的论文信息
    downloaded_papers: list[Dict[str, Any]] = field(default_factory=list)
    # 当前处理的文件路径
    current_file_path: Optional[str] = None
    # 已创建的向量数据库信息
    created_databases: list[Dict[str, Any]] = field(default_factory=list)
    # 用户原始请求
    original_query: Optional[str] = None
    # 任务状态
    task_status: str = "idle"  # idle, downloading, creating_db, querying

class AgentState(MessagesState, total=False):
    """可以路由到子代理的监督代理状态。"""
    remaining_steps: RemainingSteps
    research_context: ResearchContext  # 新增：研究上下文信息

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are a research supervisor assistant that coordinates between two specialized agents:
1. **OpenReview Agent**: Searches for academic papers from OpenReview and arXiv, and downloads them
2. **RAG Assistant**: Creates vector databases from PDFs and answers questions based on papers in the database

Today's date is {current_date}.

Your workflow for complete research tasks:
1. **Search and Download**: When users want to find and download papers, use `transfer_to_openreview_agent`
   - The agent can search OpenReview or download directly from arXiv if you know the paper
   - Downloaded papers are saved to ./data/downloads/papers/
   - **IMPORTANT**: Remember the exact filename returned by the download tool

2. **Create Vector Database**: After downloading, the RAG assistant can create a vector database from the PDF
   - Use `transfer_to_rag_assistant` with the EXACT file path returned by the download tool
   - The RAG assistant has a tool `Create_Vector_DB_From_PDF` for this purpose
   - **CRITICAL**: Use the exact filename from the download result, not a simplified version
   - Example: If download returns "[1706.03762] Attention Is All You Need_1706.03762.pdf", 
     use that exact filename, not "attention_is_all_you_need.pdf"

3. **Answer Questions**: Once the vector database is created, use `transfer_to_rag_assistant` to answer questions
   - The RAG assistant can search the vector database and answer questions based on paper content

Complete example workflow:
- User: "帮我下载 Transformer 论文并根据内容回答问题"
  1. Transfer to openreview_agent: "下载 arXiv:1706.03762 的论文"
  2. Get the exact filename from download result (e.g., "[1706.03762] Attention Is All You Need_1706.03762.pdf")
  3. Transfer to rag_assistant: "从文件 ./data/downloads/papers/[1706.03762] Attention Is All You Need_1706.03762.pdf 创建向量数据库"
  4. Transfer to rag_assistant: "根据论文内容回答：Transformer 架构的主要创新是什么？"

Available agents:
- `openreview_agent`: For searching and downloading papers from OpenReview/arXiv
- `rag_assistant`: For creating vector databases and querying paper content

Guidelines:
- **ALWAYS use the exact filename returned by download tools**
- If the user asks to search/download papers, route to openreview_agent
- If the user asks to create vector database or answer questions about papers, route to rag_assistant
- You can transfer between agents multiple times to complete complex tasks
- Always provide clear context when transferring between agents
- For well-known papers (like Transformer), you can directly instruct the agent to download from arXiv
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """用工具包装模型以进行代理转移。"""
    # Create transfer tools
    from langchain_core.tools import tool
    
    @tool
    def transfer_to_openreview_agent(query: str) -> str:
        """Transfer to OpenReview agent to search for academic papers.
        
        Use this when the user wants to:
        - Search for papers from conferences (ICML, NeurIPS, ICLR, etc.)
        - Find papers by topic, venue, or conference
        - Get paper lists or summaries
        
        Args:
            query: The user's request for searching papers
        """
        return f"Transferring to OpenReview agent with query: {query}"
    
    @tool
    def transfer_to_rag_assistant(query: str) -> str:
        """Transfer to RAG assistant to answer questions about papers in the database.
        
        Use this when the user wants to:
        - Ask questions about papers that are already in the vector database
        - Query information from stored papers
        - Get detailed answers based on paper content
        
        Args:
            query: The user's question about papers in the database
        """
        return f"Transferring to RAG assistant with query: {query}"
    
    tools = [transfer_to_openreview_agent, transfer_to_rag_assistant]
    bound_model = model.bind_tools(tools)
    
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """调用监督模型以决定路由到哪个代理。"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    
    return {"messages": [response]}


async def initialize_state(state: AgentState, config: RunnableConfig) -> AgentState:
    """初始化状态，包括研究上下文。"""
    # 初始化研究上下文（如果不存在）
    if "research_context" not in state or state["research_context"] is None:
        from dataclasses import dataclass, field
        
        @dataclass
        class ResearchContext:
            downloaded_papers: list = field(default_factory=list)
            current_file_path: str = None
            created_databases: list = field(default_factory=list)
            original_query: str = None
            task_status: str = "idle"
        
        state["research_context"] = ResearchContext()
    
    return {"messages": [], "research_context": state.get("research_context")}


async def call_openreview_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """直接复用 openreview_agent - 调用现有的 OpenReview agent 搜索论文."""
    
    # 获取最后一个 supervisor 的消息和工具调用
    last_message = state["messages"][-1]
    query = ""
    tool_call_id = None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 从工具调用中提取查询
        for tool_call in last_message.tool_calls:
            if "transfer_to_openreview_agent" in tool_call.get("name", ""):
                query = tool_call.get("args", {}).get("query", "")
                tool_call_id = tool_call.get("id")
                break
    
    # 如果没有从工具调用获取查询，使用原始用户消息
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    
    # 初始化研究上下文（如果不存在）
    if "research_context" not in state or state["research_context"] is None:
        from dataclasses import dataclass, field
        from typing import Dict, Any
        
        @dataclass
        class ResearchContext:
            downloaded_papers: list = field(default_factory=list)
            current_file_path: str = None
            created_databases: list = field(default_factory=list)
            original_query: str = None
            task_status: str = "idle"
        
        state["research_context"] = ResearchContext()
    
    research_context = state["research_context"]
    
    # 保存原始查询（如果是第一次）
    if not research_context.original_query:
        research_context.original_query = query
    
    # 更新任务状态
    research_context.task_status = "downloading"
    
    # 修复：只传递用户消息给子 agent，而不是整个状态
    from langchain_core.messages import HumanMessage
    agent_input = {
        "messages": [HumanMessage(content=query)],
        "remaining_steps": state.get("remaining_steps", 10)
    }
    
    # 使用相同的配置，确保 thread_id 等保持一致
    sub_config = RunnableConfig(
        configurable=config.get("configurable", {}),
        run_id=config.get("run_id"),
        callbacks=config.get("callbacks", []),
    )
    
    # 直接调用现有的 openreview_agent
    try:
        result = await openreview_agent.ainvoke(agent_input, sub_config)
        
        # 获取 agent 的响应
        agent_messages = result.get("messages", [])
        if agent_messages:
            # 找到最后一个 AI 消息作为响应
            last_agent_message = None
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_agent_message = msg
                    break
            
            # 改进：从响应中提取文件路径信息
            file_path = None
            if last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                
                # 尝试从内容中提取文件路径（JSON 格式或文本格式）
                import json
                import re
                
                # 方法1：尝试解析 JSON（工具返回的格式）
                try:
                    # 查找 JSON 块
                    json_match = re.search(r'\{[^{}]*"file_path"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        file_path = data.get("file_path")
                except:
                    pass
                
                # 方法2：从文本中提取路径模式
                if not file_path:
                    path_match = re.search(r'`?([./]data/downloads/papers/[^`\s]+\.pdf)`?', content)
                    if path_match:
                        file_path = path_match.group(1)
                
                # 如果找到文件路径，存储到状态中
                if file_path:
                    research_context.current_file_path = file_path
                    research_context.downloaded_papers.append({
                        "file_path": file_path,
                        "timestamp": datetime.now().isoformat()
                    })
                    research_context.task_status = "downloaded"
            
            if last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id or "unknown"
                )
                return {
                    "messages": [tool_message],
                    "research_context": research_context
                }
    except Exception as e:
        research_context.task_status = "error"
        tool_message = ToolMessage(
            content=f"调用 OpenReview agent 时出错: {str(e)}",
            tool_call_id=tool_call_id or "unknown"
        )
        return {
            "messages": [tool_message],
            "research_context": research_context
        }
    
    return {"messages": []}


async def call_rag_assistant(state: AgentState, config: RunnableConfig) -> AgentState:
    """直接复用 rag_assistant - 调用现有的 RAG assistant 回答论文相关问题."""
    
    # 获取最后一个 supervisor 的消息和工具调用
    last_message = state["messages"][-1]
    query = ""
    tool_call_id = None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 从工具调用中提取查询
        for tool_call in last_message.tool_calls:
            if "transfer_to_rag_assistant" in tool_call.get("name", ""):
                query = tool_call.get("args", {}).get("query", "")
                tool_call_id = tool_call.get("id")
                break
    
    # 如果没有从工具调用获取查询，使用原始用户消息
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    
    # 获取研究上下文
    research_context = state.get("research_context")
    
    # 改进：如果查询中包含"创建向量数据库"但没有指定文件路径，使用存储的文件路径
    if research_context and research_context.current_file_path:
        if "创建向量数据库" in query or "create vector database" in query.lower():
            # 检查查询中是否已经包含文件路径
            if research_context.current_file_path not in query:
                # 自动添加文件路径到查询中
                query = f"{query}\n\n文件路径: {research_context.current_file_path}"
                research_context.task_status = "creating_db"
    
    # 修复：只传递用户消息给子 agent，而不是整个状态
    from langchain_core.messages import HumanMessage
    agent_input = {
        "messages": [HumanMessage(content=query)],
        "remaining_steps": state.get("remaining_steps", 10)
    }
    
    # 使用相同的配置，确保 thread_id 等保持一致
    sub_config = RunnableConfig(
        configurable=config.get("configurable", {}),
        run_id=config.get("run_id"),
        callbacks=config.get("callbacks", []),
    )
    
    # 直接调用现有的 rag_assistant
    try:
        result = await rag_assistant.ainvoke(agent_input, sub_config)
        
        # 获取 agent 的响应
        agent_messages = result.get("messages", [])
        if agent_messages:
            # 找到最后一个 AI 消息作为响应
            last_agent_message = None
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_agent_message = msg
                    break
            
            # 改进：从响应中提取数据库创建信息
            if research_context and last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                
                # 检查是否成功创建了数据库
                if "创建向量数据库" in content or "vector database" in content.lower():
                    import re
                    # 尝试提取数据库路径
                    db_path_match = re.search(r'数据库路径[：:]\s*`?([^`\s]+)`?', content)
                    if db_path_match:
                        db_path = db_path_match.group(1)
                        research_context.created_databases.append({
                            "db_path": db_path,
                            "file_path": research_context.current_file_path,
                            "timestamp": datetime.now().isoformat()
                        })
                        research_context.task_status = "db_created"
            
            if last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id or "unknown"
                )
                return {
                    "messages": [tool_message],
                    "research_context": research_context
                }
    except Exception as e:
        if research_context:
            research_context.task_status = "error"
        tool_message = ToolMessage(
            content=f"调用 RAG assistant 时出错: {str(e)}",
            tool_call_id=tool_call_id or "unknown"
        )
        return {
            "messages": [tool_message],
            "research_context": research_context
        }
    
    return {"messages": []}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("initialize", initialize_state)
agent.add_node("supervisor", acall_model)
agent.add_node("openreview_agent", call_openreview_agent)
agent.add_node("rag_assistant", call_rag_assistant)

agent.set_entry_point("initialize")
agent.add_edge("initialize", "supervisor")

# 从监督者路由到适当的代理
def route_to_agent(state: AgentState) -> Literal["openreview_agent", "rag_assistant", "done"]:
    """基于工具调用路由到适当的子代理。"""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage):
        return "done"
    
    if not last_message.tool_calls:
        return "done"
    
    # Check which tool was called
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        if "transfer_to_openreview_agent" in tool_name:
            
            return "openreview_agent"
        elif "transfer_to_rag_assistant" in tool_name:
            return "rag_assistant"
    
    return "done"


agent.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "openreview_agent": "openreview_agent",
        "rag_assistant": "rag_assistant",
        "done": END,
    },
)

# 子代理完成后，返回到监督者以决定下一步操作
agent.add_edge("openreview_agent", "supervisor")
agent.add_edge("rag_assistant", "supervisor")


paper_research_supervisor = agent.compile()

