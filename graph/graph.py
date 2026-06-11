from .nodes import research_node, analyst_node, writer_node

graph = StateGraph(AgentState)

graph.add_node("researcher", research_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)

pipeline = graph.compile()