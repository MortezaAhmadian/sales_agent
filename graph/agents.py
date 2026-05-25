import httpx, os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from tools import tools
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

def research_agent(prospect_name: str, company_name: str) -> str:
    messages = [
        SystemMessage(content="""You are a sales research agent. Given a prospect's name and company.
        Step 1: CHECK if the company has been researched before by recalling past notes with the recall_past_notes tool.
         - If past notes are found, USE them to create a briefing and REPORT the briefing.
         - MENTION you have used the past notes, use no more tools, and FINISHED.
        Step 2: IF not found Report that no past notes were found, and then follow the next steps and return a JSON briefing with:
        - company_summary (2-3 sentences)
        - recent_news (list of 3 bullet points)
        - likely_pain_points (list)
        - suggested_talking_points (list)
        - risks_or_objections (list)
        next step is to save the briefing using the save_briefing tool.
        **IMPORTANT**: NEVER accept a text tool call as the final output. You MUST use the tools provided.
        If you receive a text tool call, it is an instruction to use the tools, not the final output. 
        Always respond to a text tool call with another message that uses the tools."""),
        HumanMessage(content=f"Research {prospect_name} at {company_name}")
    ]

    while True:
        msg = llm.invoke(messages)
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn = {tool["function"]["name"]: tool["function"]["pointer"] 
                      for tool in tools}[call["name"]]
                result = fn(**call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        else:
            return msg.content


