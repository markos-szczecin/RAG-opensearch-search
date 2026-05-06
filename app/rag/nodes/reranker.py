"""
LangGraph node: reranker

Reranks filtered_chunks and applies contextual compression to produce
compressed_chunks — a smaller, higher-quality set ready for token budgeting.

Reranking options (in order of complexity):
  1. Heuristic: prefer approved > draft, recent > old, higher version.
  2. Score fusion: combine retrieval score with recency + authority signals.
  3. Cross-encoder: sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2.
  4. LLM-based: ask Claude to rank chunks by relevance (expensive, accurate).

TODO (Phase 2):
  - Implement heuristic reranker as the baseline (no extra dependencies).
  - Add score boosts: +0.1 for status=approved, +0.05 per version number,
    recency bonus = days_since_valid_from / 365 * 0.1 (capped at 0.1).
  - Wire ContextualCompressor.compress() after reranking.
  - Track rerank latency separately for the debug endpoint.
"""

from app.config import get_settings
from app.rag.compressor import ContextualCompressor
from app.rag.state import RAGState

_compressor = ContextualCompressor(strategy="keyword")


async def reranker_node(state: RAGState) -> dict:
    """
    Returns compressed_chunks: reranked + compressed subset of filtered_chunks.

    TODO: implement reranking logic.
    """
    settings = get_settings()
    chunks = state.get("filtered_chunks", [])
    query = state["query"]

    # TODO: apply heuristic boosts to chunk scores
    # TODO: sort by adjusted score descending
    # TODO: keep top retrieve_top_k before compression

    compressed = await _compressor.compress(
        chunks, query, max_tokens_per_chunk=settings.max_tokens_per_chunk
    )
    return {"compressed_chunks": compressed}
