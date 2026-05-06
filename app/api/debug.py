import asyncio
import time

from fastapi import APIRouter, Depends

from app.dependencies import get_hybrid_search, get_keyword_search, get_vector_search
from app.models.search import DebugSearchResponse, SearchRequest
from app.search.base import SearchService

router = APIRouter()


@router.post("/search", response_model=DebugSearchResponse)
async def debug_search(
    request: SearchRequest,
    keyword_svc: SearchService = Depends(get_keyword_search),
    vector_svc: SearchService = Depends(get_vector_search),
    hybrid_svc: SearchService = Depends(get_hybrid_search),
) -> DebugSearchResponse:
    """
    Diagnostic endpoint that runs all three retrieval modes on the same query
    and returns their raw results plus an explanation of filtering decisions.

    Useful for:
      - Comparing keyword vs. vector vs. hybrid recall on a given query.
      - Understanding why certain chunks were discarded (access, freshness).
      - Tuning hybrid alpha, chunk size, or field boosts.

    TODO (Phase 1):
      - Run all three searches concurrently with asyncio.gather.
      - Apply RetrievalGuardrail and expose chosen vs. discarded chunks.
      - Populate notes[] with human-readable reasons for each filtering decision.
      - Add query rewriting output once the rewriting node is implemented.
    """
    start = time.monotonic()

    expanded = SearchRequest(**request.model_dump() | {"top_k": 20})
    kw_resp, vec_resp, hyb_resp = await asyncio.gather(
        keyword_svc.search(expanded),
        vector_svc.search(expanded),
        hybrid_svc.search(expanded),
    )

    latency = (time.monotonic() - start) * 1000
    return DebugSearchResponse(
        keyword_results=kw_resp.results,
        vector_results=vec_resp.results,
        hybrid_results=hyb_resp.results,
        chosen_chunks=hyb_resp.results[: request.top_k],
        discarded_chunks=[],   # TODO: populate from RetrievalGuardrail.filter_chunks()
        latency_ms=round(latency, 2),
        notes=["TODO: add filtering decision explanations"],
    )
