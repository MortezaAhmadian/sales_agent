from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from tools import tools
from state import SalesAgentState
from SalesAgent.graph.node import Node


state: SalesAgentState = {
    "prospect_name": "Elon Musk",
    "company": "Tesla",
}
research_messages = [
        SystemMessage(content="You are a research agent. Search the web and gather raw facts " \
        "about the prospect and company. Be thorough."),
        HumanMessage(content=f"Research {state['prospect_name']} at {state['company']}")
    ]

research_node = Node(tools = tools)
research_node.messages = research_messages
state = research_node.node(state)

analyst_node = Node()
analyst_node.messages = [
        SystemMessage(content="""You are a sales analyst. Given raw research, extract:
        - Top 3 pain points this company likely has
        - Product fit score (1-10) with reasoning
        - Key risks or objections to expect
        Return as structured JSON."""),
        HumanMessage(content=state["raw_research"])
    ]
state = analyst_node.node(state)
state["analysis"] = state["messages"][-1].content


writer_node = Node()
writer_node.messages = [
        SystemMessage(content="""You are a sales briefing writer. Combine the research and analysis
        into a clean, concise briefing a sales rep can read in 60 seconds.
        Format: Company snapshot, Why they need us, Talking points, Risks."""),
        HumanMessage(content=f"Research:\n{state['raw_research']}\n\nAnalysis:\n{state['analysis']}")
    ]
state = writer_node.node(state)
state["final_briefing"] = state["messages"][-1].content




graph = StateGraph(AgentState)

graph.add_node("researcher", research_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)

pipeline = graph.compile()