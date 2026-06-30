"""
LEARN: Tools are Python functions the LLM can choose to call.
The @tool decorator exposes the function name, docstring, and type hints
to the LLM so it knows when and how to use each tool.
"""
import arxiv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings

# --- Shared embedding model (loaded once, reused across tool calls) ---
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


# ─── Tool 1: arXiv ────────────────────────────────────────────────────────────

@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic research papers matching the query.
    Returns titles, authors, abstracts, and links to the papers.
    Use this when the user asks about research, papers, academic topics,
    or wants to understand the state of the art in a field.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"Abstract: {paper.summary[:400]}...\n"
            f"Link: {paper.entry_id}\n"
        )
    if not results:
        return "No papers found for this query."
    return "\n---\n".join(results)


# ─── Tool 2: Web Search ───────────────────────────────────────────────────────

# TavilySearchResults is a pre-built LangChain tool — we just configure it.
# It returns a list of {url, content} dicts; we format them as readable text.
_tavily = TavilySearchResults(
    max_results=4,
    tavily_api_key=settings.tavily_api_key,
)

@tool
def web_search(query: str) -> str:
    """
    Search the live web for recent news, articles, tutorials, or practical
    information about a topic. Use this for current events, recent developments,
    or when you need real-world examples and implementations rather than academic papers.
    """
    raw = _tavily.invoke(query)
    if not raw:
        return "No web results found."
    parts = []
    for item in raw:
        parts.append(f"URL: {item['url']}\n{item['content'][:500]}")
    return "\n---\n".join(parts)


# ─── Tool 3: Document Store (ChromaDB from Project 1) ─────────────────────────

@tool
def document_store_search(query: str) -> str:
    """
    Search through documents that the user has previously uploaded to the system.
    Use this when the question might be answered by the user's own documents,
    or when the user asks about something they uploaded. This searches a local
    vector database using semantic similarity.
    """
    vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=_get_embeddings(),
    )
    docs = vectorstore.similarity_search(query, k=settings.top_k_results)
    if not docs:
        return "No relevant documents found in the local store."
    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        results.append(f"[From: {source}]\n{doc.page_content[:400]}")
    return "\n---\n".join(results)


# Export all tools as a list — the agent graph imports this
TOOLS = [arxiv_search, web_search, document_store_search]
