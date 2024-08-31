from pydantic import BaseModel


class QuestionCreate(BaseModel):
    text: str
