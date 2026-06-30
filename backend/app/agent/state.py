"""
LEARN: AgentState is the shared memory that flows through every node in the graph.
LangGraph passes this dict from node to node, each node can read and update it.

The `add_messages` reducer is special — instead of replacing the messages list,
it *appends* new messages to it. This is how the agent builds up its conversation history.
"""
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # The user's original question — set once, never changes
    question: str
    # The running conversation (human message, AI response, tool results, ...)
    # add_messages means each update *appends* rather than *replaces*
    messages: Annotated[list, add_messages]
