from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_id: str
    topic: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    mastery: Optional[float] = None
