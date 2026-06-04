from typing import TypedDict, Annotated
import operator


class SalesAgentState(TypedDict):
    prospect_name: str
    company: str
    raw_massage: str
    analysis: str
    final_briefing: str
    messages : Annotated[list, operator.add]
