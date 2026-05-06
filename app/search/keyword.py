import time

from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.search.base import SearchService
from app.search.filters import FilterBuilder


class KeywordSearchService(SearchService):
    """
    BM25 / lexical search using OpenSearch multi_match queries.

    Best for: exact product names, policy IDs, legal phrases, acronyms.

    TODO (Phase 1 — implement the query body):
      - multi_match across ["content", "title^2"] with type="best_fields".
      - fuzziness="AUTO" for typo tolerance on content field only.
      - Add a phrase match (match_phrase) as a should clause to boost
        documents that contain the exact phrase.
      - Enable highlighting on the content field (fragment_size=200, 1 fragment).
      - Support pagination via request.from_ and request.top_k.
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        index_name: str,
        settings: Settings,
    ) -> None:
        self._client = client
        self._index = index_name
        self._filter_builder = FilterBuilder(settings)

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.monotonic()

        query_body = self._build_query(request)

        # TODO: execute query
        # response = await self._client.search(index=self._index, body=query_body)
        # results = self._parse_response(response)

        latency = (time.monotonic() - start) * 1000
        # TODO: return real results
        return SearchResponse(
            results=[],
            total=0,
            retrieval_mode="keyword",
            latency_ms=round(latency, 2),
        )

    def _build_query(self, request: SearchRequest) -> dict:
        """
        Construct the full OpenSearch query dict.

        TODO: implement multi_match + filter + highlight.
        """
        filter_clause = self._filter_builder.build(request.filters, request.user_role)

        return {
            "size": request.top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["content", "title^2"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                # TODO: add phrase_prefix / match_phrase should clause
                            }
                        }
                    ],
                    "filter": filter_clause,
                }
            },
            "highlight": {
                "fields": {"content": {"fragment_size": 200, "number_of_fragments": 1}}
            },
        }

    def _parse_response(self, response: dict) -> list[SearchResult]:
        """
        TODO: map OpenSearch hits to SearchResult objects.
        Extract highlight snippet into SearchResult.highlight.
        """
        raise NotImplementedError
