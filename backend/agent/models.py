from pydantic import BaseModel

class BasicResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]



