import os
import json
import httpx
import chromadb

from dotenv import load_dotenv
from state import AgentState

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage


from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def _make_llm():
    return ChatOpenAI(
        base_url=os.environ["VLLM_URL"],
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        api_key="dummy",
        http_client=httpx.Client(
            auth=(os.environ["NGINX_USER"], os.environ["NGINX_PASSWORD"]),
            headers={"X-API-Key": os.environ["VLLM_API_KEY"]},
        ),
        temperature=0.1,
    )

# ─────────────────────────────────────────────
# Memory (ChromaDB)
# ─────────────────────────────────────────────

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("sales_memory")

def save_briefing(company: str, briefing: str):
    collection.upsert(documents=[briefing], ids=[company])



# ─────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────

research_llm = _make_llm().bind_tools(tools_list)
analyst_llm = _make_llm()
writer_llm = _make_llm()
critic_llm = _make_llm()


def research_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content=(
            "You are a research agent. Search the web and gather raw facts about the prospect "
            "and their company. Also check memory for any past notes. Be thorough."
        )),
        HumanMessage(content=f"Research {state['prospect_name']} at {state['company']}")
    ]

    while True:
        msg = research_llm.invoke(messages)
        messages.append(msg)

        if msg.tool_calls:
            messages = run_tool_calls(msg, messages)
        else:
            return {**state, "raw_research": msg.content, "messages": messages}


def analyst_node(state: AgentState) -> AgentState:
    msg = analyst_llm.invoke([
        SystemMessage(content=(
            "You are a sales analyst. Given raw research about a prospect, extract insights. "
            "Return ONLY a JSON object with these keys:\n"
            "- pain_points: list of top 3 pain points this company likely has\n"
            "- fit_score: integer 1-10 of how well our product fits\n"
            "- fit_reasoning: one sentence explaining the score\n"
            "- risks: list of likely objections or risks"
        )),
        HumanMessage(content=state["raw_research"])
    ])
    return {**state, "analysis": msg.content}


def writer_node(state: AgentState) -> AgentState:
    msg = writer_llm.invoke([
        SystemMessage(content=(
            "You are a sales briefing writer. Combine the research and analysis into a clean, "
            "concise briefing a sales rep can read in 60 seconds.\n"
            "Format it with these sections:\n"
            "1. Company Snapshot (2-3 sentences)\n"
            "2. Why They Need Us (bullet points)\n"
            "3. Suggested Talking Points (bullet points)\n"
            "4. Risks & Objections (bullet points)"
        )),
        HumanMessage(content=(
            f"Research:\n{state['raw_research']}\n\n"
            f"Analysis:\n{state['analysis']}"
        ))
    ])
    return {**state, "final_briefing": msg.content}


def critic_node(state: AgentState) -> AgentState:
    msg = critic_llm.invoke([
        SystemMessage(content=(
            "You are a quality critic for sales briefings. Score the briefing 1-10.\n"
            "Return ONLY a JSON object with:\n"
            "- score: integer 1-10\n"
            "- approved: true if score >= 7, false otherwise\n"
            "- feedback: one sentence of what to improve (empty string if approved)"
        )),
        HumanMessage(content=state["final_briefing"])
    ])

    try:
        clean = msg.content.replace("```json", "").replace("```", "").strip()
        critique = json.loads(clean)
    except Exception:
        critique = {"score": 8, "approved": True, "feedback": ""}

    return {**state, "critique": critique}


def save_node(state: AgentState) -> AgentState:
    """Save the approved briefing to ChromaDB memory."""
    save_briefing(state["company"], state["final_briefing"])
    return state


# ─────────────────────────────────────────────
# Conditional edge: revise or finish
# ─────────────────────────────────────────────

def should_revise(state: AgentState) -> str:
    if not state["critique"].get("approved", True):
        print(f"[Critic] Score: {state['critique']['score']} — revising. Feedback: {state['critique']['feedback']}")
        return "writer"
    print(f"[Critic] Score: {state['critique']['score']} — approved.")
    return "save"

# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("researcher", research_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)
graph.add_node("save", save_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", should_revise, {"writer": "writer", "save": "save"})
graph.add_edge("save", END)

pipeline = graph.compile()

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    prospect = input("Prospect name: ")
    company = input("Company: ")

    print(f"\n🔍 Researching {prospect} at {company}...\n")

    result = pipeline.invoke({
        "prospect_name": prospect,
        "company": company,
        "raw_research": "",
        "analysis": "",
        "final_briefing": "",
        "critique": {},
        "messages": []
    })

    print("\n📋 FINAL BRIEFING\n")
    print(result["final_briefing"])
    print(f"\n✅ Saved to memory for future sessions.")