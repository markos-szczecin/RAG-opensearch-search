from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Optional metadata filters applied before or alongside retrieval."""

    language: str | None = None
    doc_type: str | None = None
    access_level: str | None = None     # overrides role-derived access if set explicitly
    status: str | None = "approved"     # default: only approved documents
    department: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=10, ge=1, le=50)
    user_role: str = Field(default="customer", description="Drives access_level filter")


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    score: float
    source_path: str
    access_level: str
    doc_type: str
    highlight: str | None = None    # populated by keyword search highlighting


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    retrieval_mode: Literal["keyword", "vector", "hybrid"]
    latency_ms: float


class DebugSearchResponse(BaseModel):
    """
    Extended response for /debug/search. Exposes internals to aid relevance debugging.
    All three retrieval modes are run and their raw outputs exposed.
    """

    rewritten_query: str | None = None   # TODO: populate after adding query rewriting node
    keyword_results: list[SearchResult]
    vector_results: list[SearchResult]
    hybrid_results: list[SearchResult]
    chosen_chunks: list[SearchResult]    # chunks that passed permission + rerank
    discarded_chunks: list[SearchResult] # chunks retrieved but filtered out
    latency_ms: float
    notes: list[str] = Field(default_factory=list)  # human-readable explanation of decisions
