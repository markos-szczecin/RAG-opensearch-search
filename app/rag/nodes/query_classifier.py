"""
LangGraph node: query_classifier

Classifies the incoming query into one of four categories:
  - retrieval   → proceed to retrieve node
  - smalltalk   → short-circuit to safe_direct_answer
  - unsafe      → short-circuit to refusal
  - unclear     → short-circuit to clarification

TODO (Phase 2):
  - Call Anthropic claude-haiku-4-5 (cheap, fast) with QUERY_CLASSIFIER_PROMPT.
  - Parse the single-word response; default to "retrieval" on unexpected output.
  - Add a fast regex pre-filter for known injection patterns BEFORE the LLM call
    (e.g. "ignore previous instructions") so unsafe queries never reach the LLM.
  - Log the classification result and confidence for analytics.
"""

from app.rag.state import RAGState


async def query_classifier_node(state: RAGState) -> dict:
    """
    Returns a partial RAGState dict with query_class populated.

    TODO: replace stub with real classification.
    """
    query = state["query"]

    # --- Fast-path regex checks (implement before LLM call) ---
    # TODO: check INJECTION_PATTERNS from guardrails/input.py
    # if InputGuardrail().check(query).action == "block":
    #     return {"query_class": "unsafe"}

    # --- LLM classification (TODO) ---
    # response = await anthropic_client.messages.create(...)
    # query_class = response.content[0].text.strip().lower()

    # Stub: treat everything as needing retrieval
    return {"query_class": "retrieval"}


def route_query(state: RAGState) -> str:
    """Routing function for LangGraph conditional edges."""
    return state.get("query_class", "retrieval")
