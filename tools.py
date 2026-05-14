from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests, json

def web_search(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    return json.dumps([{"title": r["title"], "body": r["body"]} for r in results])

def scrape_web(url: str) -> str:
    html = requests.get(url, timeout=10)
    soup = BeautifulSoup(html.text, 'html.parser')
    return soup.get_text(separator=" ", strip=True)[:3000]


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
    }
]