from ddgs import DDGS
from bs4 import BeautifulSoup
import requests, json
import chromadb


_CHROMA = chromadb.Client()
_COLLECTION = _CHROMA.get_or_create_collection("sales_agent")

def web_search(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    return json.dumps([{"title": r["title"], "body": r["body"]} for r in results])

def scrape_page(url: str) -> str:
    html = requests.get(url, timeout=10)
    soup = BeautifulSoup(html.text, 'html.parser')
    return soup.get_text(separator=" ", strip=True)[:3000]

def save_briefing(name: str, briefing: str) -> None:
    _COLLECTION.add(
        documents=[briefing],
        ids=[name]
    )

def recall_past_notes(company: str) -> str:
    results = _COLLECTION.query(
        query_texts=[company],
        n_results=1
    )
    return results["documents"][0][0] if results["documents"] else "No past notes found."


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information about a company or person",
            "parameters": 
                {
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
            "parameters": 
                {
                    "type": "object", 
                    "properties": {"url": {"type": "string"}}, 
                    "required": ["url"]
                }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_briefing",
            "description": "Save a briefing for a company",
            "parameters": 
                {
                    "type": "object", 
                    "properties": {"name": {"type": "string"}, "briefing": {"type": "string"}}, 
                    "required": ["name", "briefing"]
                }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_past_notes",
            "description": "Recall past notes about a company",
            "parameters": 
                {
                    "type": "object", 
                    "properties": {"company": {"type": "string"}}, 
                    "required": ["company"]
                }
        }
    }
]