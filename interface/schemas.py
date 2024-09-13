from typing import List

from pydantic import BaseModel


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
