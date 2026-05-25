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
    """
    Jeden wynik wyszukiwania — reprezentuje fragment (chunk) dokumentu.

    Dlaczego przechowujemy tutaj i chunk_id i doc_id?
    ---------------------------------------------------
    chunk_id identyfikuje konkretny fragment (np. "policy-001::chunk-002"),
    natomiast doc_id identyfikuje cały dokument źródłowy ("policy-001").
    Do deduplikacji w hybrid search używamy chunk_id (chcemy unikalne fragmenty).
    Do cytowania w odpowiedzi używamy doc_id (cytujemy dokument, nie fragment).

    Dlaczego status i version są opcjonalne (None)?
    ------------------------------------------------
    Pola te są przydatne w rerankerze (faworyzujemy approved > draft,
    wyższa wersja > starsza), ale nie wszystkie ścieżki wyszukiwania
    mapują te pola. Opcjonalność pozwala stopniowo wzbogacać wyniki
    bez łamania istniejących wywołań. Reranker sprawdza `if chunk.status`.
    """

    chunk_id: str
    doc_id: str
    title: str
    content: str
    score: float
    source_path: str
    access_level: str
    doc_type: str
    highlight: str | None = None    # wypełniane przez keyword search highlighting
    status: str | None = None       # "approved" | "draft" — do rerankeringu
    version: int | None = None      # numer wersji dokumentu — do rerankeringu


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
