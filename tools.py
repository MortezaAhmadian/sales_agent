from ddgs import DDGS
from bs4 import BeautifulSoup
import requests, json
import chromadb


_CHROMA = chromadb.PersistentClient(path="./chroma_db")
_COLLECTION = _CHROMA.get_or_create_collection("sales_agent")

def web_search(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    return json.dumps([{"title": r["title"], "body": r["body"]} for r in results])

def scrape_page(url: str) -> str:
    html = requests.get(url, timeout=10)
    soup = BeautifulSoup(html.text, 'html.parser')
    return soup.get_text(separator=" ", strip=True)[:3000]

def save_briefing(company: str, briefing: str) -> None:
    _COLLECTION.add(
        documents=[briefing],
        ids=[company]
    )

def recall_past_notes(company: str) -> str:
    results = _COLLECTION.query(
        query_texts=[company],
        n_results=1
    )
    try:
        past_notes = results["documents"][0][0]
    except IndexError as e:
        past_notes = f"No past notes found for {company}."
    except Exception as e:
        past_notes = f"Error occurred: {e}"
    return past_notes


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information about a company or person",
            "pointer": web_search,
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
            "pointer": scrape_page,
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
            "pointer": save_briefing,
            "parameters": 
                {
                    "type": "object", 
                    "properties": {"company": {"type": "string"}, "briefing": {"type": "string"}}, 
                    "required": ["company", "briefing"]
                }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_past_notes",
            "description": "Recall past notes about a company",
            "pointer": recall_past_notes,
            "parameters": 
                {
                    "type": "object", 
                    "properties": {"company": {"type": "string"}}, 
                    "required": ["company"]
                }
        }
    }
]