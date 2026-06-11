from langchain_core.messages import SystemMessage, HumanMessage

research_llm = _make_llm().bind_tools([web_search_tool, scrape_tool, recall_tool])
analyst_llm = _make_llm()
writer_llm = _make_llm()

def research_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="You are a research agent. Search the web and gather raw facts about the prospect and company. Be thorough."),
        HumanMessage(content=f"Research {state['prospect_name']} at {state['company']}")
    ]
    # run the ReAct tool loop
    while True:
        msg = research_llm.invoke(messages)
        messages.append(msg)
        if msg.tool_calls:
            for call in msg.tool_calls:
                fn = {"web_search": web_search, "scrape_page": scrape_page, "recall_past_notes": recall_past_notes}[call["name"]]
                result = fn(**call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        else:
            return {**state, "raw_research": msg.content, "messages": messages}

def analyst_node(state: AgentState) -> AgentState:
    msg = analyst_llm.invoke([
        SystemMessage(content="""You are a sales analyst. Given raw research, extract:
        - Top 3 pain points this company likely has
        - Product fit score (1-10) with reasoning
        - Key risks or objections to expect
        Return as structured JSON."""),
        HumanMessage(content=state["raw_research"])
    ])
    return {**state, "analysis": msg.content}

def writer_node(state: AgentState) -> AgentState:
    msg = writer_llm.invoke([
        SystemMessage(content="""You are a sales briefing writer. Combine the research and analysis
        into a clean, concise briefing a sales rep can read in 60 seconds.
        Format: Company snapshot, Why they need us, Talking points, Risks."""),
        HumanMessage(content=f"Research:\n{state['raw_research']}\n\nAnalysis:\n{state['analysis']}")
    ])
    # save to memory
    save_briefing(state["company"], msg.content)
    return {**state, "final_briefing": msg.content}