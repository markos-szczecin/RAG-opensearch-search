from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal[1, 2, 3, 4, 5]
    comment: str | None = None
    user_role: str = "customer"
    retrieval_mode: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str = "Feedback recorded. Thank you."
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
