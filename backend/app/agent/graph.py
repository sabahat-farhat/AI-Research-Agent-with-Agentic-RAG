"""
LEARN: This file defines the agent as a LangGraph StateGraph.
A StateGraph is a directed graph where:
- Nodes are Python functions that receive and return AgentState
- Edges define what runs next
- Conditional edges let the LLM's output decide the next path

The ReAct loop we build here:
  agent_node → (has tool calls?) → tool_node → agent_node → ... → END
"""
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.tools import TOOLS
from app.core.config import settings


def build_agent_graph():
    # LLM with tools bound — this tells the LLM what tools exist and their schemas.
    # "bind_tools" adds the tool definitions to every request so the LLM can choose to call them.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # bigger model = better tool-use reasoning
        temperature=0,
        api_key=settings.groq_api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    # ─── Node 1: Agent ────────────────────────────────────────────────────────
    # This node runs the LLM. The LLM sees all messages so far and either:
    #   (a) calls a tool  → adds an AIMessage with tool_calls
    #   (b) gives a final answer → adds an AIMessage with content
    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # ─── Node 2: Tools ────────────────────────────────────────────────────────
    # ToolNode is a prebuilt LangGraph node. It:
    #   1. Reads tool_calls from the last AIMessage
    #   2. Executes the matching Python function from TOOLS
    #   3. Wraps the result in a ToolMessage and appends it to messages
    tool_node = ToolNode(TOOLS)

    # ─── Conditional edge: should we loop or stop? ────────────────────────────
    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        # If the LLM's last message contains tool calls → run those tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        # Otherwise the LLM gave a final text answer → stop
        return "end"

    # ─── Build the graph ──────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    # After agent runs, decide: loop to tools, or end
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"call_tools": "tools", "end": END},
    )

    # After tools run, always go back to agent (it decides what to do next)
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compile once at import time — the compiled graph is thread-safe and reusable
agent_graph = build_agent_graph()
