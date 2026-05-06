from abc import ABC, abstractmethod

from app.models.search import SearchRequest, SearchResponse


class SearchService(ABC):
    """
    Single-method interface for all retrieval modes.

    Follows Interface Segregation: callers only depend on search(), not on
    how embedding, query building, or score normalization are done internally.
    """

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a search and return ranked results.

        Implementations must:
          - Apply metadata filters from request.filters via FilterBuilder.
          - Honour request.user_role for access-level filtering.
          - Return results with the correct retrieval_mode label.
          - Record latency in the response.
        """
        ...
