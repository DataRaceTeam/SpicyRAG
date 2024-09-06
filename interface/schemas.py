from pydantic import BaseModel


class QuestionCreate(BaseModel):
    question: str
