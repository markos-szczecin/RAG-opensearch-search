from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_hybrid_search, get_keyword_search, get_vector_search
from app.models.search import SearchRequest, SearchResponse
from app.search.base import SearchService

router = APIRouter()


def _search_router(service: SearchService):
    """Factory: returns a route handler bound to the given search service."""

    async def handler(request: SearchRequest) -> SearchResponse:
        """
        TODO (Phase 1):
          - Run request through InputGuardrail first.
          - Call service.search(request).
          - Log result to EvalRepository.
          - Return SearchResponse.
        """
        return await service.search(request)

    return handler


@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(
    request: SearchRequest,
    service: SearchService = Depends(get_keyword_search),
) -> SearchResponse:
    """BM25 lexical search. Best for exact terms and product names."""
    return await service.search(request)


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    service: SearchService = Depends(get_vector_search),
) -> SearchResponse:
    """Semantic kNN search. Best for paraphrased or conceptual queries."""
    return await service.search(request)


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    service: SearchService = Depends(get_hybrid_search),
) -> SearchResponse:
    """Hybrid BM25 + vector search. Recommended default for fintech."""
    return await service.search(request)
