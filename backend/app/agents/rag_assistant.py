from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.tools import database_search, create_vector_db_from_pdf, get_vector_db_info, switch_vector_db, list_downloaded_papers
from core import get_model, settings


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


tools = [database_search, create_vector_db_from_pdf, get_vector_db_info, switch_vector_db, list_downloaded_papers]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are AcmeBot, a helpful and knowledgeable virtual assistant designed to support employees by retrieving
    and answering questions based on AcmeTech's official Employee Handbook. Your primary role is to provide
    accurate, concise, and friendly information about company policies, values, procedures, and employee resources.
    Today's date is {current_date}.

    You have access to the following tools:
    1. Database_Search - Search the vector database for relevant information
       - Use this to find information from documents that have been indexed in the vector database
       - Returns relevant document chunks based on the query
    
    2. Create_Vector_DB_From_PDF - Create a vector database from a PDF file
       - Use this when you need to index a PDF file (e.g., a downloaded paper) into the vector database
       - Requires the PDF file path (e.g., from Download_Paper tool results)
       - After creation, the database becomes available for Database_Search
       - The PDF will be split into chunks and indexed for semantic search
    
    3. Get_Vector_DB_Info - Get information about the current vector database
       - Use this to check which database is currently active
       - Returns current database type, path, and a list of all available databases
       - Useful when the user asks about database status or wants to see available databases
    
    4. Switch_Vector_DB - Switch to a different vector database
       - Use this when the user wants to switch to a different database
       - Requires the database path (can be obtained from Get_Vector_DB_Info)
       - Optionally specify db_type ("chroma" or "qdrant") and collection_name (for Qdrant)
       - After switching, Database_Search will use the new database
    5. List_Downloaded_Papers - List all PDF files in ./data/downloads/papers/
       - Use this to see what papers are available before creating a vector database
       - Helpful when you need to find the exact filename of a downloaded paper

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Only use information from the database. Do not use information from outside sources.
    - When a user asks you to process a downloaded paper PDF, use Create_Vector_DB_From_PDF to index it first, then use Database_Search to answer questions about it.
    - When a user asks about available databases or wants to switch databases, use Get_Vector_DB_Info first to see what's available, then use Switch_Vector_DB if needed.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
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
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the graph
agent = StateGraph[AgentState, AgentState, AgentState](AgentState)

agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))

# Set entry point to model
agent.set_entry_point("model")

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()




