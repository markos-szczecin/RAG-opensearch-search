"""
Information retrieval evaluation metrics.

All functions operate on lists of doc_ids (strings).  They are pure functions
with no side effects so they are trivially testable.

Reference:
  - Manning et al. "Introduction to Information Retrieval", Chapter 8.
  - https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

TODO (Phase 4):
  - Add NDCG with graded relevance labels (not just binary relevant/irrelevant).
  - Add latency_percentiles() helper for p50/p95/p99 from a list of latencies.
  - Add token_cost_estimate(input_tokens, output_tokens, model) using published
    Anthropic pricing so the evaluation script can report estimated $ cost.
"""


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Fraction of the top-k retrieved documents that are relevant.

    P@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved: Ordered list of retrieved doc_ids (rank 1 first).
        relevant:  Set of ground-truth relevant doc_ids.
        k:         Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Fraction of relevant documents that appear in the top-k results.

    R@k = |retrieved[:k] ∩ relevant| / |relevant|

    Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(top_k & relevant_set) / len(relevant_set)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """
    Mean Reciprocal Rank for a single query.

    MRR = 1 / rank_of_first_relevant_document
    Returns 0.0 if no relevant document appears in retrieved.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k (binary relevance).

    DCG@k  = Σ rel_i / log2(i + 1)   for i in 1..k
    IDCG@k = DCG of ideal ranking (all relevant docs first)
    NDCG@k = DCG@k / IDCG@k

    Returns 0.0 if relevant is empty or k <= 0.

    TODO (Phase 4): extend to graded relevance (rel_i ∈ {0, 1, 2, 3}).
    """
    import math

    if not relevant or k <= 0:
        return 0.0

    relevant_set = set(relevant)

    def dcg(ranked: list[str]) -> float:
        return sum(
            (1.0 if doc_id in relevant_set else 0.0) / math.log2(i + 2)
            for i, doc_id in enumerate(ranked[:k])
        )

    ideal = ["relevant"] * min(len(relevant), k)   # all relevant first
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal)))
    actual_dcg = dcg(retrieved)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(retrieved: list[str], relevant: list[str]) -> float:
    """
    Average Precision for a single query (used to compute MAP across queries).

    AP = (1/|R|) * Σ P@k * rel_k   for k in 1..len(retrieved)
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / k
    return precision_sum / len(relevant_set)
