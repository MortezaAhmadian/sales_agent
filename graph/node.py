import httpx, os
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
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
        temperature=0.0,
    )


class Node:
    def __init__(self, tools: list = None):
        self.tools = tools
        if self.tools is None:
            self.llm = _make_llm().bind_tools(tools)
        self.messages: list = []
        self.name: str = ""
    
    def node(self, name: str, messages: list) -> str:
        self.name = name
        self.messages = messages
        if not self.tools:
            msg = self.llm.invoke(self.messages)
            self.messages.append(msg)
            return self.messages

        while True:
            msg = self.llm.invoke(self.messages)
            self.messages.append(msg)

            if msg.tool_calls:
                for call in msg.tool_calls:
                    fn = {tool["function"]["name"]: tool["function"]["pointer"] 
                        for tool in self.tools}[call["name"]]
                    result = fn(**call["args"])
                    self.messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            else:
                return self.name, self.messages
    