"""Sales agent LangGraph pipeline.

Flow:  researcher → analyst → writer → critic → (revise? writer : save) → END

The researcher has three tools:
  1. query_knowledge_base  — RAG over pgvector (called first, read-only)
  2. web_search            — DuckDuckGo
  3. scrape_page           — webpage text extraction

Approved briefings are printed to stdout. RAG is used for retrieval only.
"""

import json
import logging
import os

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from rag_store import RAGStore
from state import AgentState
from tools import run_tool_calls, set_rag_store, tools_list

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── RAG store — initialised once at import time ────────────────────────────

rag_store = RAGStore()
set_rag_store(rag_store)
logger.info("RAG store ready — knowledge chunks: %d", rag_store.knowledge_count)

# ── LLM factory ───────────────────────────────────────────────────────────


def _make_llm() -> ChatOpenAI:
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


# ── LLMs — one per role ───────────────────────────────────────────────────

research_llm = _make_llm().bind_tools(tools_list)
analyst_llm  = _make_llm()
writer_llm   = _make_llm()
critic_llm   = _make_llm()

# ── Nodes ─────────────────────────────────────────────────────────────────


def research_node(state: AgentState) -> AgentState:
    """Gather raw facts using RAG + web search."""
    messages = [
        SystemMessage(content=(
            "You are a sales research agent. Use the tools in this exact order:\n"
            "1. query_knowledge_base — search internal docs and past briefings first.\n"
            "2. web_search — find recent news about the company and prospect.\n"
            "3. scrape_page — read the most relevant URLs from search results.\n"
            "Use all three before concluding. Be thorough."
        )),
        HumanMessage(content=f"Research {state['prospect_name']} at {state['company']}"),
    ]

    while True:
        msg = research_llm.invoke(messages)
        messages.append(msg)

        if msg.tool_calls:
            messages = run_tool_calls(msg, messages)
        else:
            return {**state, "raw_research": msg.content, "messages": messages}


def analyst_node(state: AgentState) -> AgentState:
    """Extract structured insights from raw research."""
    msg = analyst_llm.invoke([
        SystemMessage(content=(
            "You are a sales analyst. Given raw research about a prospect, extract insights.\n"
            "Return ONLY a JSON object with these keys:\n"
            "- pain_points: list of top 3 pain points this company likely has\n"
            "- fit_score: integer 1-10 of how well our product fits\n"
            "- fit_reasoning: one sentence explaining the score\n"
            "- risks: list of likely objections or risks"
        )),
        HumanMessage(content=state["raw_research"]),
    ])
    return {**state, "analysis": msg.content}


def writer_node(state: AgentState) -> AgentState:
    """Write the sales briefing."""
    # If the critic gave feedback, include it so the writer can improve.
    feedback = state["critique"].get("feedback", "")
    feedback_section = f"\n\nCritic feedback to address:\n{feedback}" if feedback else ""

    msg = writer_llm.invoke([
        SystemMessage(content=(
            "You are a sales briefing writer. Combine the research and analysis into a clean,\n"
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
            f"{feedback_section}"
        )),
    ])
    return {**state, "final_briefing": msg.content}


def critic_node(state: AgentState) -> AgentState:
    """Score the briefing and decide whether it needs revision."""
    msg = critic_llm.invoke([
        SystemMessage(content=(
            "You are a quality critic for sales briefings. Score the briefing 1-10.\n"
            "Return ONLY a JSON object with:\n"
            "- score: integer 1-10\n"
            "- approved: true if score >= 7, false otherwise\n"
            "- feedback: one sentence of what to improve (empty string if approved)"
        )),
        HumanMessage(content=state["final_briefing"]),
    ])

    try:
        clean = msg.content.replace("```json", "").replace("```", "").strip()
        critique = json.loads(clean)
    except Exception:
        # Default to approved if parsing fails — don't block on a bad parse
        critique = {"score": 8, "approved": True, "feedback": ""}

    return {**state, "critique": critique}


def save_node(state: AgentState) -> AgentState:
    """Log completion. RAG is read-only; no memory write."""
    logger.info("Briefing for '%s' approved and complete.", state["company"])
    return state
    return state


# ── Conditional edge ──────────────────────────────────────────────────────


def should_revise(state: AgentState) -> str:
    approved = state["critique"].get("approved", True)
    score    = state["critique"].get("score", "?")
    feedback = state["critique"].get("feedback", "")

    if not approved:
        logger.info("Critic score %s — revising. Feedback: %s", score, feedback)
        return "writer"

    logger.info("Critic score %s — approved.", score)
    return "save"


# ── Graph ─────────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("researcher", research_node)
graph.add_node("analyst",    analyst_node)
graph.add_node("writer",     writer_node)
graph.add_node("critic",     critic_node)
graph.add_node("save",       save_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst",    "writer")
graph.add_edge("writer",     "critic")
graph.add_conditional_edges("critic", should_revise, {"writer": "writer", "save": "save"})
graph.add_edge("save", END)

pipeline = graph.compile()

# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    prospect = input("Prospect name: ").strip()
    company  = input("Company:        ").strip()

    print(f"\n🔍 Researching {prospect} at {company}...\n")

    result = pipeline.invoke({
        "prospect_name":  prospect,
        "company":        company,
        "raw_research":   "",
        "analysis":       "",
        "final_briefing": "",
        "critique":       {},
        "messages":       [],
    })

    print("\n📋 FINAL BRIEFING\n")
    print(result["final_briefing"])

    rag_store.close()
