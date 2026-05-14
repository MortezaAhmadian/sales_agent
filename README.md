# Sales Research Agent

An agentic AI that researches prospects before sales calls.
Given a name + company, it searches the web, scrapes relevant pages,
and returns a structured briefing in seconds.

## Setup
1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI key
4. `python agent.py`

## Stack
- OpenAI GPT-4o (ReAct agent loop)
- DuckDuckGo Search
- BeautifulSoup