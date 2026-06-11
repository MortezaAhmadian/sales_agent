import collections

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import json

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from collections import collection


def recall_past_notes(company: str) -> str:
    try:
        results = collection.query(query_texts=[company], n_results=1)
        docs = results["documents"]
        if docs and docs[0]:
            return docs[0][0]
        return "No prior history."
    except Exception:
        return "No prior history."

def web_search(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    return json.dumps([{"title": r["title"], "body": r["body"]} for r in results])

def scrape_page(url: str) -> str:
    try:
        html = requests.get(url, timeout=8).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape page: {str(e)}"

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information about a company or person",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Fetch and read the content of a webpage",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_past_notes",
            "description": "Recall past briefings or notes about a company from memory",
            "parameters": {
                "type": "object",
                "properties": {"company": {"type": "string"}},
                "required": ["company"]
            }
        }
    }
]

tool_fn_map = {
    "web_search": web_search,
    "scrape_page": scrape_page,
    "recall_past_notes": recall_past_notes,
}

def run_tool_calls(msg, messages: list) -> list:
    """Execute all tool calls in a message and append results."""
    for call in msg.tool_calls:
        fn = tool_fn_map[call["name"]]
        result = fn(**call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return messages