"""All tools available to the sales agent.

One source of truth. No duplicates.

Tools exposed to the LLM:
  - web_search           search DuckDuckGo for recent news
  - scrape_page          fetch and parse a webpage
  - query_knowledge_base semantic search over the RAG knowledge base (read-only)

Helpers used internally by the graph:
  - run_tool_calls       dispatch LLM tool calls to their Python functions
"""

import json
import logging

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# The RAGStore is injected at startup via set_rag_store().
# Tools call it through this module-level reference.
_rag_store = None


def set_rag_store(store) -> None:
    """Call once at startup, before the graph runs."""
    global _rag_store
    _rag_store = store


# ── Tool implementations ───────────────────────────────────────────────────


def web_search(query: str) -> str:
    """Search DuckDuckGo and return the top 5 results as JSON."""
    try:
        results = DDGS().text(query, max_results=5)
        return json.dumps([{"title": r["title"], "body": r["body"]} for r in results])
    except Exception as exc:
        logger.warning("web_search failed: %s", exc)
        return json.dumps([])


def scrape_page(url: str) -> str:
    """Fetch a webpage and return up to 3 000 chars of visible text."""
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as exc:
        return f"Could not scrape page: {exc}"


def query_knowledge_base(query: str, component: str = "", doc_type: str = "", k: int = 5) -> str:
    """Semantic search over the pgvector knowledge base.

    Always call this FIRST before web_search to check whether we already have
    internal knowledge about this company, industry, or topic.

    Args:
        query:     Natural-language search query.
        component: Optional filter (e.g. "nar", "freddiemac").
        doc_type:  Optional filter (e.g. "tutorial", "docs").
        k:         Number of results to return (default 5).
    """
    if _rag_store is None:
        return "RAG store not initialised — results unavailable."

    results = _rag_store.query(
        query_text=query,
        k=k,
        component=component or None,
        doc_type=doc_type or None,
    )

    if not results:
        return f"No RAG results found for: '{query}'"

    lines = [f"=== RAG results for '{query}' ({len(results)} chunks) ===\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("source", "unknown")
        lines.append(f"[{i}] source={source}  score={r['score']}")
        lines.append(r["content"])
        lines.append("")
    return "\n".join(lines)


# ── LangChain tool schemas ─────────────────────────────────────────────────
# These dicts are passed to bind_tools() so the LLM knows the tool signatures.

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": (
                "Semantic search over internal knowledge docs. "
                "ALWAYS call this first before web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "What to search for"},
                    "component": {"type": "string", "description": "Optional source filter"},
                    "doc_type":  {"type": "string", "description": "Optional doc type filter"},
                    "k":         {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent news about a company or person.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Fetch and read the content of a webpage URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

# ── Dispatcher ────────────────────────────────────────────────────────────

_TOOL_FN_MAP = {
    "web_search":           web_search,
    "scrape_page":          scrape_page,
    "query_knowledge_base": query_knowledge_base,
}


def run_tool_calls(msg, messages: list) -> list:
    """Execute every tool call in `msg` and append ToolMessage results."""
    for call in msg.tool_calls:
        fn = _TOOL_FN_MAP.get(call["name"])
        if fn is None:
            result = f"Unknown tool: {call['name']}"
        else:
            try:
                result = fn(**call["args"])
            except Exception as exc:
                result = f"Tool error: {exc}"
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return messages
