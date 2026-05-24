import httpx, os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from tools import web_search, scrape_page, save_briefing, recall_past_notes, tools
from dotenv import load_dotenv

load_dotenv()

def _make_llm():
    return ChatOpenAI(
        base_url=os.environ["VLLM_URL"],
        model=os.getenv("VLLM_MODEL"),
        api_key="dummy",
        http_client=httpx.Client(
            auth=(os.environ["NGINX_USER"], os.environ["NGINX_PASSWORD"]),
            headers={"X-API-Key": os.environ["VLLM_API_KEY"]},
        ),
        temperature=0.1,
    )

llm = _make_llm().bind_tools(tools)

