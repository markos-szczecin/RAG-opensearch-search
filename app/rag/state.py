from typing import TypedDict

from app.models.rag import Citation, TokenUsage
from app.models.search import SearchResult


class RAGState(TypedDict, total=False):
    """
    Shared state object passed between every LangGraph node.

    Each node reads what it needs and returns a dict with only the keys it
    modifies — LangGraph merges the partial dict into the running state.

    Field lifecycle:
      query / user_role / retrieval_mode / answer_mode / chat_history
        → set by the entry point, never mutated

      query_class       → set by query_classifier node
      raw_chunks        → set by retrieve node
      filtered_chunks   → set by permission_filter node
      compressed_chunks → set by reranker / compressor node
      budgeted_chunks   → set by context_budgeter node
      answer            → set by answer_generator node
      citations         → set by answer_generator node
      confidence        → set by answer_validator node
      tokens            → set by answer_generator, updated by context_budgeter
      error             → set by any node that encounters a non-recoverable error
    """

    # ---- Input (set by API layer) ----
    query: str
    user_role: str
    retrieval_mode: str       # keyword | vector | hybrid
    answer_mode: str          # brief | detailed | step_by_step | agent
    chat_history: list[dict]

    # ---- Classification ----
    query_class: str          # smalltalk | unsafe | unclear | retrieval

    # ---- Retrieval pipeline ----
    raw_chunks: list[SearchResult]          # straight from the search service
    filtered_chunks: list[SearchResult]     # after permission guardrail
    compressed_chunks: list[SearchResult]   # after reranker / compressor
    budgeted_chunks: list[SearchResult]     # after token budgeting

    # ---- Generation ----
    answer: str
    citations: list[Citation]
    confidence: str           # grounded | cautious | refused
    tokens: TokenUsage

    # ---- Error propagation ----
    error: str | None
