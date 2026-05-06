"""
LangGraph node: permission_filter

Filters raw_chunks to only those the user is authorised to see, based on:
  - access_level (public / internal / confidential) mapped from user_role
  - document status (draft chunks excluded unless explicitly opted in)
  - valid_to date (expired documents hidden from LLM context)

This is the retrieval-layer guardrail — it runs AFTER search but BEFORE the
LLM ever sees document content, so no unauthorised text leaks into prompts.

TODO (Phase 3):
  - Import RetrievalGuardrail and call it per chunk.
  - Log discarded chunks with reason codes for the debug endpoint.
  - Add a "soft filter" mode that includes expired docs with a warning label
    (useful for compliance officers reviewing historical policies).
"""

from datetime import date

from app.config import get_settings
from app.rag.state import RAGState


async def permission_filter_node(state: RAGState) -> dict:
    """
    Returns filtered_chunks: subset of raw_chunks that pass access checks.

    TODO: implement access_level + freshness filtering.
    """
    settings = get_settings()
    user_role = state.get("user_role", "customer")
    raw = state.get("raw_chunks", [])

    # TODO: get allowed access levels from settings.role_access_levels[user_role]
    # TODO: filter by access_level, status, and valid_to date

    # Stub: pass all chunks through
    return {"filtered_chunks": raw}
