"""
LangGraph node: context_budgeter

Enforces the token budget before the expensive LLM generation call.

Token budget policy (from Settings):
  max_context_tokens  = 2500   total tokens for retrieved context
  max_chunks          = 5      hard cap on chunk count
  max_tokens_per_chunk = 400   per-chunk content limit

Algorithm:
  1. Count system_prompt + chat_history tokens.
  2. Iterate compressed_chunks in score order; add each until budget exhausted.
  3. Trim the last included chunk's content if it would overflow the budget.

TODO (Phase 2):
  - Use tiktoken (cl100k_base) to count tokens accurately.
  - Account for CONTEXT_BLOCK_TEMPLATE overhead (~30 tokens per chunk).
  - Expose remaining_budget in state so answer_generator can set max_tokens.
  - Log dropped chunks (chunk_id + reason "budget_exceeded") for debug endpoint.
"""

import tiktoken

from app.config import get_settings
from app.models.rag import TokenUsage
from app.rag.state import RAGState

_enc = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_enc.encode(text))


async def context_budgeter_node(state: RAGState) -> dict:
    """
    Returns budgeted_chunks + initial TokenUsage with context token count.

    TODO: implement proper token counting and trimming.
    """
    settings = get_settings()
    chunks = state.get("compressed_chunks", [])

    # Stub: take up to max_chunks without counting tokens
    budgeted = chunks[: settings.max_chunks]

    context_tokens = sum(_count(c.content) for c in budgeted)
    return {
        "budgeted_chunks": budgeted,
        "tokens": TokenUsage(context=context_tokens, answer=0, total=context_tokens),
    }
