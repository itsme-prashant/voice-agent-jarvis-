from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # The list of messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The next agent to route to, or "FINISH"
    next: str
