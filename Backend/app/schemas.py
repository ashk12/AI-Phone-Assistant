from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    intent: str
    confidence: float
    response_text: str
