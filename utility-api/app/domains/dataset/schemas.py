from pydantic import BaseModel


class QuestionOut(BaseModel):
    """Single question returned by the API."""

    query: str

    model_config = {"from_attributes": True}


class NextQuestionResponse(BaseModel):
    """Response for GET /dataset/next."""

    epoch_id: int
    uid: int
    search_type: str
    question: QuestionOut
