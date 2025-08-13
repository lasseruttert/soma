from typing import TypedDict, List, Optional, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[List[dict], add]
    next_agent: Optional[str]
    session_id: str
    metadata: dict