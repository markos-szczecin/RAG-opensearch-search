"""
LangGraph node: answer_validator

Validates the generated answer for groundedness before returning it.

Groundedness levels:
  grounded     → all major claims map to retrieved chunks; return as-is
  cautious     → some claims are weakly supported; prepend a caveat
  unsupported  → answer contradicts or ignores the evidence; route to refusal

Validation strategies:
  1. Citation check: at least one [doc_id] reference present (fast, cheap)
  2. Keyword overlap: key entities in the answer appear in retrieved chunks
  3. LLM judge: ask Claude to score groundedness (expensive but accurate)

TODO (Phase 3):
  - Implement citation_check() — fail if no citations found.
  - Implement keyword_overlap_check() — extract named entities from the answer,
    verify each appears in at least one budgeted chunk.
  - Optionally add LLM judge with GROUNDEDNESS_CHECK_PROMPT for high-stakes queries.
  - Route "cautious" answers: prepend
    "Based on available documents (confidence: cautious): " to the answer.
"""

from app.rag.prompts import GROUNDEDNESS_CHECK_PROMPT
from app.rag.state import RAGState


async def answer_validator_node(state: RAGState) -> dict:
    """
    Returns confidence level.  In stub mode, always returns 'grounded'.

    TODO: implement real validation checks.
    """
    answer = state.get("answer", "")
    citations = state.get("citations", [])

    # Fast check 1: citations present
    if not citations:
        return {"confidence": "cautious"}

    # TODO: keyword overlap check
    # TODO: optional LLM judge

    return {"confidence": "grounded"}


def route_validation(state: RAGState) -> str:
    """Routing function for LangGraph conditional edges after validation."""
    confidence = state.get("confidence", "grounded")
    if confidence == "unsupported":
        return "refusal"
    return "end"   # both grounded and cautious go to final answer
