# Research Agent — Agentic RAG with Real-Time Streaming

A multi-tool AI research agent that autonomously decides how to answer a question, searching arXiv papers, the live web, or a local document store while streaming its reasoning process to a React frontend in real time.

---

## What It Does

Ask the agent a research question. Instead of a single LLM call, it runs a **ReAct loop**:

1. **Reason** — the LLM decides which tool will best answer the question
2. **Act** — it calls that tool (arXiv, web search, or your uploaded documents)
3. **Observe** — it reads the result and decides whether it needs more information
4. **Repeat** until it has enough context, then generates a final answer

Every step of this process streams to the browser live — you watch the agent think and act in real time.

---

## Demo

```
User: "What are the latest advances in RAG systems?"

🤔  "I'll search arXiv for recent papers on RAG"
🔧  arXiv Search — "retrieval augmented generation 2024"
      → Found 5 papers: "RAPTOR: Recursive...", "Self-RAG...", ...

🤔  "Let me also check for practical implementations on the web"
🔧  Web Search — "RAG systems production 2025"
      → Found articles from LangChain blog, Towards Data Science...

✅  Final Answer
      RAG (Retrieval Augmented Generation) has seen major advances in...
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Agent framework | LangGraph | Models the ReAct loop as a state graph |
| LLM | Groq (LLaMA 3.3 70B) | Reasons and decides which tools to use |
| Web search | Tavily API | Searches the live web for current information |
| Paper search | arXiv Python library | Searches 2M+ academic research papers |
| Vector store | ChromaDB (from Project 1) | Semantic search over uploaded documents |
| Backend | FastAPI + SSE | Streams agent events to the frontend in real time |
| Frontend | React + Vite + Tailwind | Renders the agent's thought process live |

---

## Project Structure

```
03-research-agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app with SSE streaming endpoint
│   │   ├── agent/
│   │   │   ├── graph.py         # LangGraph agent — nodes, edges, ReAct loop
│   │   │   ├── tools.py         # 3 tools: arXiv, web search, document store
│   │   │   └── state.py         # AgentState — shared memory across graph nodes
│   │   └── core/
│   │       └── config.py        # Settings loaded from .env
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── AgentStream.jsx  # Manages SSE connection, renders events live
│           ├── ToolCallBadge.jsx # Shows which tool ran and its result
│           └── FinalAnswer.jsx  # Renders the agent's final response
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- A **Groq** API key — free at [console.groq.com](https://console.groq.com)
- A **Tavily** API key — free tier at [tavily.com](https://tavily.com) (1000 searches/month)

---

## Setup

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your .env file
copy .env.example .env    # Windows
cp .env.example .env      # macOS / Linux
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Start the server:

```bash
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## How It Works — Key Concepts

### LangGraph State Machine

The agent is modelled as a directed graph with two nodes:

```
START → [agent node] ──── has tool calls? ────► [tool node] ──┐
              ▲                                                  │
              └──────────────────────────────────────────────────┘
              │
         no tool calls → END
```

- **agent node** — runs the LLM. The LLM either calls a tool or generates a final answer.
- **tool node** — LangGraph's `ToolNode` automatically executes whichever tool the LLM requested and appends the result back to the message history.
- **conditional edge** — `should_continue()` inspects the LLM's last message. If it contains tool calls, loop to `tool_node`. If not, stop.

### Tools

Each tool is a plain Python function decorated with `@tool`. The decorator extracts the function's name, docstring, and type hints to build a schema the LLM uses to decide when and how to call it.

| Tool | Source | Best for |
|---|---|---|
| `arxiv_search` | arXiv API (free, no key) | Academic papers, research, state of the art |
| `web_search` | Tavily API | Current news, tutorials, practical guides |
| `document_store_search` | ChromaDB (local) | Questions about your own uploaded documents |


---

