import time

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_graph
from app.guardrails.input import InputGuardrail
from app.models.rag import AskRequest, AskResponse, TokenUsage

router = APIRouter()

_input_guardrail = InputGuardrail()


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    rag_graph=Depends(get_rag_graph),
) -> AskResponse:
    """
    RAG question-answering endpoint.

    Pipeline:
      1. InputGuardrail check (fast, sync)
      2. Invoke LangGraph workflow
      3. OutputGuardrail check
      4. Return AskResponse with citations + token usage

    TODO (Phase 2):
      - Wire real InputGuardrail result → 400 if blocked.
      - Pass chat_history into graph state for multi-turn support.
      - Log request + response stats to EvalRepository.
      - Add streaming support (FastAPI StreamingResponse) for long answers.
    """
    start = time.monotonic()

    # Step 1: input guardrail
    guardrail_result = _input_guardrail.check(request.query)
    if guardrail_result.action == "block":
        raise HTTPException(
            status_code=400,
            detail=guardrail_result.reason or "Query blocked by safety guardrail",
        )

    # Step 2: invoke graph
    initial_state = {
        "query": guardrail_result.safe_query or request.query,
        "user_role": request.user_role,
        "retrieval_mode": request.retrieval_mode,
        "answer_mode": request.answer_mode,
        "chat_history": request.chat_history,
        "error": None,
    }

    # TODO: await rag_graph.ainvoke(initial_state)
    # final_state = await rag_graph.ainvoke(initial_state)
    # For now, return a stub response.
    raise HTTPException(status_code=501, detail="RAG graph not yet implemented — Phase 2")
