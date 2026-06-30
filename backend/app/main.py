"""
LEARN: This is the FastAPI entry point for the Research Agent.

Key new concept: StreamingResponse with Server-Sent Events (SSE).
Instead of waiting for the full answer and returning JSON, we:
1. Start the agent
2. As each step completes, immediately send an event to the browser
3. The browser renders each event as it arrives — the user sees the agent think live

SSE format (what we send over the wire):
    data: {"type": "thinking", "content": "..."}\n\n
    data: {"type": "tool_call", "tool": "arxiv_search", "args": {...}}\n\n
    data: {"type": "tool_result", "content": "..."}\n\n
    data: {"type": "final_answer", "content": "..."}\n\n
    data: {"type": "done"}\n\n
"""
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agent.graph import agent_graph

app = FastAPI(title="Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str


def sse_event(data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"data: {json.dumps(data)}\n\n"


async def run_agent_stream(question: str):
    """
    LEARN: This async generator runs the LangGraph agent and yields SSE events.

    .astream_events() is a LangGraph method that fires events as the graph runs.
    Each event has a "name" (what happened) and "data" (the payload).

    We listen for 3 event types:
    - "on_chat_model_end" → the LLM finished a response (thinking or final answer)
    - "on_tool_start"     → a tool is about to be called
    - "on_tool_end"       → a tool finished and returned a result
    """
    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
    }

    try:
        async for event in agent_graph.astream_events(initial_state, version="v2"):
            event_name = event.get("name", "")
            event_type = event.get("event", "")

            # LLM finished generating a response
            if event_type == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output is None:
                    continue
                # Get the AIMessage from the output
                if hasattr(output, "content") and output.content:
                    # Check if this is a tool call decision or final answer
                    has_tool_calls = (
                        hasattr(output, "tool_calls") and bool(output.tool_calls)
                    )
                    if has_tool_calls:
                        # LLM decided to call tools — share its reasoning
                        yield sse_event({
                            "type": "thinking",
                            "content": output.content,
                        })
                    else:
                        # LLM gave a final answer — no more tool calls
                        yield sse_event({
                            "type": "final_answer",
                            "content": output.content,
                        })

            # A tool is about to run — tell the UI which tool and with what args
            elif event_type == "on_tool_start":
                tool_name = event.get("name", "unknown_tool")
                tool_input = event.get("data", {}).get("input", {})
                yield sse_event({
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_input,
                })

            # A tool finished — send a preview of its result
            elif event_type == "on_tool_end":
                tool_name = event.get("name", "unknown_tool")
                output = event.get("data", {}).get("output", "")
                # Truncate long tool outputs — the LLM gets the full version,
                # but we only show a preview in the UI
                preview = str(output)[:600] + "..." if len(str(output)) > 600 else str(output)
                yield sse_event({
                    "type": "tool_result",
                    "tool": tool_name,
                    "content": preview,
                })

        # Signal to the frontend that the stream is complete
        yield sse_event({"type": "done"})

    except Exception as e:
        yield sse_event({"type": "error", "content": str(e)})
        yield sse_event({"type": "done"})


@app.post("/research")
async def research(request: ResearchRequest):
    """
    SSE streaming endpoint. Returns a text/event-stream response.

    Why StreamingResponse instead of a normal return?
    A normal FastAPI endpoint buffers the full response before sending.
    StreamingResponse sends data as it's produced — essential for live streaming.
    """
    if not request.question.strip():
        return {"error": "Question cannot be empty"}

    return StreamingResponse(
        run_agent_stream(request.question),
        media_type="text/event-stream",
        headers={
            # Disable buffering at the proxy/CDN level
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
