from app.models.search import SearchResult


class ContextualCompressor:
    """
    Extracts query-relevant sentences from each retrieved chunk.

    Purpose: reduce prompt tokens without discarding evidence.  Instead of
    sending 400-token chunks verbatim, extract only the 50–100 tokens that
    directly answer the query.

    Strategies (in increasing accuracy / cost order):
      1. Sentence-level keyword overlap (baseline, no API call)
      2. Cross-encoder relevance scoring (sentence-transformers cross-encoder)
      3. LLM extraction — ask Claude to return only relevant sentences

    TODO (Phase 3 — pick one strategy and implement it):
      Strategy 1 (free):
        - Tokenise chunk into sentences (spaCy or simple regex).
        - Score each sentence by n-gram overlap with the query.
        - Keep top-K sentences; reconstruct as a coherent passage.

      Strategy 2 (local model):
        - Load cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
        - Score (query, sentence) pairs; keep sentences above threshold.
        - More accurate than keyword overlap; no API cost.

      Strategy 3 (LLM):
        - Build a prompt: "Return only the sentences relevant to: {query}".
        - Fast because we use claude-haiku-4-5 for this step, not sonnet.
        - Adds latency + cost; worthwhile only if chunk quality is poor.
    """

    def __init__(self, strategy: str = "keyword") -> None:
        self._strategy = strategy

    async def compress(
        self,
        chunks: list[SearchResult],
        query: str,
        max_tokens_per_chunk: int = 200,
    ) -> list[SearchResult]:
        """
        Return new SearchResult objects with content trimmed to relevant excerpts.
        Scores are preserved from the original retrieval.

        TODO: implement chosen strategy.
        """
        # Passthrough stub — returns chunks unchanged
        return chunks
