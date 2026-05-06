"""
LangGraph node: answer_generator

Calls Anthropic Claude to generate a grounded answer from budgeted_chunks.

Implementation notes:
  - Use prompt caching on the system prompt and context block to reduce cost
    on repeated queries to the same documents.
  - Parse the response to extract citations: look for [doc_id] patterns and
    map them back to the corresponding SearchResult objects.
  - Record actual token usage from response.usage to update state.tokens.

TODO (Phase 2):
  - Instantiate AsyncAnthropic client (inject via config, not hardcoded).
  - Build the context_block string from CONTEXT_BLOCK_TEMPLATE * n chunks.
  - Add cache_control={'type': 'ephemeral'} to system + large context blocks.
  - Handle anthropic.APIError with a graceful error message (set state.error).
  - Support answer_mode by appending a style instruction to the prompt.
  - For step_by_step mode, ask Claude to number each step explicitly.
"""

import re

from app.models.rag import Citation, TokenUsage
from app.rag.prompts import (
    ANSWER_GENERATOR_PROMPT,
    CONTEXT_BLOCK_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.rag.state import RAGState


def _build_context_block(state: RAGState) -> str:
    chunks = state.get("budgeted_chunks", [])
    parts = [
        CONTEXT_BLOCK_TEMPLATE.format(
            index=i + 1,
            title=c.title,
            doc_id=c.doc_id,
            content=c.content,
        )
        for i, c in enumerate(chunks)
    ]
    return "\n".join(parts)


def _extract_citations(answer: str, state: RAGState) -> list[Citation]:
    """
    Parse [doc_id] references from the answer text and match them to chunks.

    TODO: handle partial matches (e.g. user writes [mobile-auth] instead of
    full doc_id).  Consider normalising doc_ids to lowercase for matching.
    """
    chunks_by_doc = {c.doc_id: c for c in state.get("budgeted_chunks", [])}
    cited_ids = re.findall(r"\[([^\]]+)\]", answer)
    citations = []
    for doc_id in dict.fromkeys(cited_ids):   # preserve order, deduplicate
        if doc_id in chunks_by_doc:
            c = chunks_by_doc[doc_id]
            citations.append(
                Citation(
                    doc_id=c.doc_id,
                    title=c.title,
                    chunk_id=c.chunk_id,
                    score=c.score,
                    source_path=c.source_path,
                )
            )
    return citations


async def answer_generator_node(state: RAGState) -> dict:
    """
    TODO: replace stub with real Anthropic API call.

    Stub returns a placeholder answer so the graph compiles end-to-end.
    """
    context_block = _build_context_block(state)

    # TODO: build messages list with cache_control breakpoints
    # messages = [
    #     {
    #         "role": "user",
    #         "content": ANSWER_GENERATOR_PROMPT.format(
    #             context_block=context_block, query=state["query"]
    #         ),
    #     }
    # ]
    # response = await client.messages.create(
    #     model=settings.llm_model,
    #     max_tokens=settings.max_answer_tokens,
    #     system=[{"type": "text", "text": SYSTEM_PROMPT.format(...), "cache_control": {...}}],
    #     messages=messages,
    # )
    # answer = response.content[0].text
    # usage = response.usage

    answer = (
        "[STUB] Answer generation not yet implemented. "
        "See answer_generator.py for TODO instructions."
    )
    existing_tokens = state.get("tokens", TokenUsage(context=0, answer=0, total=0))
    answer_tokens = len(answer.split())   # rough estimate; replace with response.usage
    total = existing_tokens.context + answer_tokens

    return {
        "answer": answer,
        "citations": _extract_citations(answer, state),
        "tokens": TokenUsage(
            context=existing_tokens.context, answer=answer_tokens, total=total
        ),
    }
