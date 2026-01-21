from dataclasses import dataclass
# 类型别名，用于处理LangGraph的不同代理模式
from langgraph.graph.state import CompiledStateGraph
# Pregel是一个可以并行执行的图
from langgraph.pregel import Pregel

from agents.openreview_agent import openreview_agent
from agents.paper_research_supervisor import paper_research_supervisor
from agents.rag_assistant import rag_assistant
from schema import AgentInfo
# LazyLoadingAgent是需要异步加载的代理的基类
from agents.lazy_agent import LazyLoadingAgent

DEFAULT_AGENT = "rag-assistant"

AgentGraph = CompiledStateGraph | Pregel 

AgentGraphLike = CompiledStateGraph | Pregel | LazyLoadingAgent


@dataclass
class Agent:
    description: str
    graph_like : AgentGraphLike


agents: dict[str, Agent] = {
    "rag-assistant": Agent(
        description="一个可以访问数据库中信息的RAG助手。",
        graph_like=rag_assistant,
    ),
    "openreview-agent": Agent(
        description="专门从OpenReview（ICML、NeurIPS、ICLR等）搜索学术论文的代理。",
        graph_like=openreview_agent,
    ),
    "paper-research-supervisor": Agent(
        description="一个监督代理，协调OpenReview代理（用于搜索论文）和RAG助手（用于查询论文数据库）。可以搜索论文并回答相关问题。",
        graph_like=paper_research_supervisor,
    ),

}


async def load_agent(agent_id: str) -> None:
    """如果需要，加载延迟代理。"""
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraphLike:
    """获取代理图，如果需要则加载延迟代理。"""
    agent_graph = agents[agent_id].graph_like

    # 如果是延迟加载代理，确保它已加载并返回其图
    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")
        return agent_graph.get_graph()

    # 否则直接返回图
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]