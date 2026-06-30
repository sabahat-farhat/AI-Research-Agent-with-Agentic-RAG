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
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.tools import TOOLS
from app.core.config import settings

SYSTEM_PROMPT = """You are a research assistant with access to three tools:
- arxiv_search: search academic papers on arXiv
- web_search: search the live web for recent articles and news
- document_store_search: search documents the user has uploaded locally

For each user question:
1. Decide which tool(s) would best answer it
2. Call the tool with a clear, specific query
3. After receiving results, decide if you need another tool or have enough to answer
4. When you have enough information, give a thorough final answer

Always call at least one tool before answering. Never make up information — base your answer only on tool results."""


def build_agent_graph():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    # Fallback LLM without tools — used if the model fails to generate a valid tool call
    llm_plain = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_api_key,
    )

    def agent_node(state: AgentState):
        # Prepend the system prompt if this is the first turn
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        try:
            response = llm_with_tools.invoke(messages)
            # Groq sometimes returns an AIMessage with empty content AND no tool calls
            # when it fails to form a valid tool call — treat that as a plain answer
            has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)
            has_content = bool(response.content and response.content.strip())
            if not has_tool_calls and not has_content:
                raise ValueError("Empty response from model")
            return {"messages": [response]}
        except Exception:
            # If tool-call generation fails, fall back to a direct answer
            fallback = llm_plain.invoke(messages)
            return {"messages": [AIMessage(content=fallback.content)]}

    tool_node = ToolNode(TOOLS)

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"call_tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


agent_graph = build_agent_graph()
