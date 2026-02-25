from datetime import datetime

from pydantic import BaseModel


class QuestionOut(BaseModel):
    """Single question returned by the API."""

    query: str
    params: dict = {}

    model_config = {"from_attributes": True}


class NextQuestionResponse(BaseModel):
    """Response for GET /dataset/next."""

    time_range_start: datetime
    uid: int
    search_type: str
    question: QuestionOut
