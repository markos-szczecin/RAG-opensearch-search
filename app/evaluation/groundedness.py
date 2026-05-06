"""
Answer groundedness checker.

Checks whether every factual claim in the generated answer is supported by
at least one retrieved chunk.

Strategies:
  1. Citation coverage (fast): verify each [doc_id] reference appears in chunks.
  2. N-gram overlap (medium): key entities in the answer overlap with chunk text.
  3. NLI model (slow, accurate): use a cross-encoder trained on NLI tasks.

TODO (Phase 4):
  - Implement _check_citation_coverage(): scan answer for [doc_id] patterns,
    verify each referenced doc_id is in the provided chunks.
  - Implement _check_entity_overlap(): use spaCy to extract named entities from
    the answer, verify each appears in at least one chunk.
  - Add a per-sentence breakdown so evaluation scripts can report which
    sentences are grounded and which are not.
"""

import re

from app.models.rag import Citation
from app.models.search import SearchResult


class GroundednessChecker:
    """
    Evaluates whether a generated answer is grounded in retrieved chunks.

    Returns a score in [0, 1] and a list of ungrounded claims.
    """

    def check(
        self,
        answer: str,
        chunks: list[SearchResult],
        citations: list[Citation],
    ) -> dict:
        """
        Returns:
        {
          "score": float,           # 0.0 (fully ungrounded) to 1.0 (fully grounded)
          "grounded": bool,
          "ungrounded_claims": list[str],
          "missing_citations": list[str],
        }

        TODO: implement all checks.  Stub returns fully grounded.
        """
        chunk_ids = {c.chunk_id for c in chunks}
        cited_ids = re.findall(r"\[([^\]]+)\]", answer)

        missing = [cid for cid in cited_ids if cid not in chunk_ids]

        return {
            "score": 1.0 if not missing else max(0.0, 1.0 - len(missing) / max(len(cited_ids), 1)),
            "grounded": len(missing) == 0,
            "ungrounded_claims": [],       # TODO: implement sentence-level check
            "missing_citations": missing,
        }
