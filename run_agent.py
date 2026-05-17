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

def run_agent(prospect_name: str, company_name: str) -> str:
    messages = [
        SystemMessage(content="""You are a sales research agent. Given a prospect's name and company,
        use your tools to research them thoroughly. Then return a JSON briefing with:
        - company_summary (2-3 sentences)
        - recent_news (list of 3 bullet points)
        - likely_pain_points (list)
        - suggested_talking_points (list)
        - risks_or_objections (list)"""),
        HumanMessage(content=f"Research {prospect_name} at {company_name}")
    ]

    while True:
        msg = llm.invoke(messages)
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn = {"web_search": web_search, "scrape_page": scrape_page}[call["name"]]
                result = fn(**call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        else:
            return msg.content

if __name__ == "__main__":
    briefing = run_agent("Elon Musk ", "Tesla")
    print(briefing)