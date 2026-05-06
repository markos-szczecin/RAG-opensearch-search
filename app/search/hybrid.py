import asyncio
import time

from app.config import Settings
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.search.base import SearchService
from app.search.keyword import KeywordSearchService
from app.search.vector import VectorSearchService


class HybridSearchService(SearchService):
    """
    Combines keyword (BM25) and vector (kNN) results using score fusion.

    Hybrid retrieval is the recommended default for fintech because:
      - BM25 handles exact product names, IDs, and legal phrases well.
      - Vectors handle paraphrased queries and conceptual similarity.
      - Combining both improves recall without sacrificing precision.

    Score fusion approach:
      1. Run both searches concurrently (asyncio.gather).
      2. Normalise each result set's scores to [0, 1] via min-max scaling.
      3. Merge: combined_score = alpha * bm25_norm + (1 - alpha) * vec_norm
      4. Deduplicate by chunk_id, keeping the highest combined score.
      5. Sort descending, return top_k.

    TODO (Phase 1):
      - Implement _normalise_scores() with min-max scaling; guard against
        all-equal scores (return 0.5 for all to avoid division by zero).
      - Implement _merge() with the deduplication logic above.
      - Expose alpha as a request-level override (not just config) so callers
        can tune keyword vs. vector weight per query.

    TODO (Phase 2 advanced):
      - Try Reciprocal Rank Fusion (RRF) as an alternative to linear fusion;
        it is rank-based so scores don't need normalisation.
    """

    def __init__(
        self,
        keyword_service: KeywordSearchService,
        vector_service: VectorSearchService,
        alpha: float,
        settings: Settings,
    ) -> None:
        self._keyword = keyword_service
        self._vector = vector_service
        self._alpha = alpha          # weight for BM25; (1 - alpha) for vector
        self._settings = settings

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.monotonic()

        # Expand candidate pool before merging; top_k is the final result count
        expanded = SearchRequest(
            **request.model_dump() | {"top_k": self._settings.retrieve_top_k}
        )
        kw_resp, vec_resp = await asyncio.gather(
            self._keyword.search(expanded),
            self._vector.search(expanded),
        )

        kw_norm = self._normalise_scores(kw_resp.results)
        vec_norm = self._normalise_scores(vec_resp.results)
        merged = self._merge(kw_norm, vec_norm)[: request.top_k]

        latency = (time.monotonic() - start) * 1000
        return SearchResponse(
            results=merged,
            total=len(merged),
            retrieval_mode="hybrid",
            latency_ms=round(latency, 2),
        )

    def _normalise_scores(self, results: list[SearchResult]) -> dict[str, tuple[SearchResult, float]]:
        """
        Returns {chunk_id: (result, normalised_score)}.

        TODO: implement min-max normalisation.
        """
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s or 1.0   # guard against identical scores
        return {r.chunk_id: (r, (r.score - min_s) / span) for r in results}

    def _merge(
        self,
        kw: dict[str, tuple[SearchResult, float]],
        vec: dict[str, tuple[SearchResult, float]],
    ) -> list[SearchResult]:
        """
        Fuse normalised keyword and vector scores.

        TODO: implement combined scoring and deduplication.
        """
        combined: dict[str, tuple[SearchResult, float]] = {}
        all_ids = set(kw) | set(vec)
        for chunk_id in all_ids:
            kw_result, kw_score = kw.get(chunk_id, (None, 0.0))
            vec_result, vec_score = vec.get(chunk_id, (None, 0.0))
            result = (kw_result or vec_result)
            if result is None:
                continue
            fused = self._alpha * kw_score + (1 - self._alpha) * vec_score
            combined[chunk_id] = (result, fused)

        return [r for r, _ in sorted(combined.values(), key=lambda x: x[1], reverse=True)]
