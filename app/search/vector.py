import time

from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.indexing.embedder.base import Embedder
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.search.base import SearchService
from app.search.filters import FilterBuilder


class VectorSearchService(SearchService):
    """
    Semantic / approximate nearest-neighbour search using OpenSearch kNN.

    Best for: conceptually related content even when exact words differ.

    TODO (Phase 1):
      - Embed the query with self._embedder.embed_query().
      - Build a knn query targeting the content_vector field.
      - Apply efficient_filter (pre-filtering) to avoid scanning irrelevant access levels.
      - Measure embedding latency separately from index latency for profiling.
      - Experiment with ef_search parameter for recall/latency trade-off.
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        embedder: Embedder,
        index_name: str,
        settings: Settings,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._index = index_name
        self._filter_builder = FilterBuilder(settings)
        self._settings = settings

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.monotonic()

        # Step 1: embed the query
        query_vector = await self._embedder.embed_query(request.query)

        # Step 2: build knn query body
        query_body = self._build_query(request, query_vector)

        # Step 3: execute (TODO)
        # response = await self._client.search(index=self._index, body=query_body)
        # results = self._parse_response(response)

        latency = (time.monotonic() - start) * 1000
        return SearchResponse(
            results=[],
            total=0,
            retrieval_mode="vector",
            latency_ms=round(latency, 2),
        )

    def _build_query(self, request: SearchRequest, query_vector: list[float]) -> dict:
        """
        TODO: implement knn query with efficient_filter.
        efficient_filter applies the metadata filter *before* the ANN search,
        reducing candidates and improving both performance and accuracy.
        """
        filter_clause = self._filter_builder.build(request.filters, request.user_role)

        return {
            "size": request.top_k,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": query_vector,
                        "k": request.top_k,
                        # TODO: "efficient_filter": filter_clause,
                    }
                }
            },
            # TODO: apply filter_clause as a post-filter if efficient_filter unsupported
        }

    def _parse_response(self, response: dict) -> list[SearchResult]:
        """TODO: map knn hits to SearchResult objects."""
        raise NotImplementedError
