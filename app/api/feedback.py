import uuid

from fastapi import APIRouter, Depends

from app.dependencies import get_eval_repository
from app.evaluation.repository import EvalRepository
from app.models.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    repo: EvalRepository = Depends(get_eval_repository),
) -> FeedbackResponse:
    """
    Collect user ratings (1–5 stars) and optional comments on RAG answers.

    Feedback is stored in PostgreSQL and used to:
      - Identify queries where retrieval quality is poor.
      - Track answer satisfaction over time.
      - Prioritise documents for reindexing or review.

    TODO (Phase 4):
      - Call repo.log_feedback(request) once implemented.
      - Validate that rating is 1–5 (Pydantic already enforces this).
      - Consider adding a "was this helpful?" boolean field as an
        even simpler UX signal alongside the star rating.
    """
    # TODO: feedback_id = await repo.log_feedback(request)
    feedback_id = str(uuid.uuid4())   # stub
    return FeedbackResponse(feedback_id=feedback_id)
