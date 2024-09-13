from pydantic import BaseModel
from typing import List, Dict, Any


class QuestionCreate(BaseModel):
    question: str


class EmbedderSettings(BaseModel):
    batch_size: int = 16
    model_name: str
    model_type: str
    dimension: int
    prefix_query: str
    prefix_document: str


class QuestionResponse(BaseModel):
    response: str
    context: List[str]
