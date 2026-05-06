from fastapi import APIRouter

from app.api import ask, debug, documents, feedback, health, search

router = APIRouter()

router.include_router(health.router, tags=["Health"])
router.include_router(documents.router, prefix="/documents", tags=["Indexing"])
router.include_router(search.router, prefix="/search", tags=["Search"])
router.include_router(ask.router, tags=["RAG"])
router.include_router(feedback.router, tags=["Feedback"])
router.include_router(debug.router, prefix="/debug", tags=["Debug"])
