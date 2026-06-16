from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    prospect_name: str
    company: str
    raw_research: str
    analysis: str
    final_briefing: str
    critique: dict
    messages: Annotated[list, operator.add]
