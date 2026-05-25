"""
Endpoint /ask — główny interfejs RAG question-answering.

Przepływ żądania:
  1. InputGuardrail: szybka weryfikacja zapytania (regex injection, długość, PII)
  2. LangGraph ainvoke: pełny pipeline RAG (classifier → retrieve → rank → generate)
  3. Mapowanie stanu grafu na AskResponse
  4. Opcjonalnie: logowanie do bazy danych (EvalRepository)

Dlaczego guardrail jest PRZED grafem?
---------------------------------------
InputGuardrail to synchroniczny check regex — zajmuje < 1ms i nie wymaga API.
Uruchomienie go przed grafem oszczędza:
  - Embedding zapytania (~100ms)
  - OpenSearch search (~10ms)
  - Klasyfikację haiku (~100ms)
  - Generowanie odpowiedzi (~1000ms)
  ...dla zapytań które i tak zostaną zablokowane.

Dla ataków injection (często automatycznych botów) to 1200ms zaoszczędzone per żądanie.

Dlaczego nie logujemy w tym endpointzie?
------------------------------------------
EvalRepository.log_search() zapisuje do PostgreSQL — wywołanie await może dodać
10-50ms do latency odpowiedzi. Rozwiązanie: fire-and-forget przez asyncio.create_task()
lub logowanie asynchroniczne przez queue. Na etapie MVP pominięte dla prostoty.
"""

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
    Odpowiada na pytanie bazując na indeksowanych dokumentach.

    Zwraca odpowiedź z cytowaniami, poziomem pewności i statystykami tokenów.

    Args:
        request:    AskRequest z zapytaniem, rolą użytkownika i trybami.
        rag_graph:  Skompilowany LangGraph graph (singleton z DI).

    Returns:
        AskResponse z:
          - answer: tekst odpowiedzi (może zawierać [doc_id] cytowania inline)
          - citations: lista Citation z metadanymi cytowanych dokumentów
          - confidence: "grounded" | "cautious" | "refused"
          - tokens: TokenUsage z kosztem tokenów kontekstu i odpowiedzi
          - latency_ms: całkowita latency pipeline'u

    Raises:
        HTTPException 400: gdy zapytanie jest zablokowane przez InputGuardrail
        HTTPException 500: gdy LangGraph rzuci nieoczekiwany wyjątek
    """
    start = time.monotonic()

    # ------------------------------------------------------------------
    # Krok 1: InputGuardrail (< 1ms, synchroniczny)
    # ------------------------------------------------------------------
    guardrail_result = _input_guardrail.check(request.query)
    if guardrail_result.action == "block":
        raise HTTPException(
            status_code=400,
            detail=guardrail_result.reason or "Query blocked by safety guardrail",
        )

    # Jeśli guardrail wykrył PII (action="warn"), używamy zredagowanego zapytania
    # do wyszukiwania i generowania, ale NIE logujemy oryginalnego zapytania.
    safe_query = guardrail_result.safe_query or request.query

    # ------------------------------------------------------------------
    # Krok 2: Invoke LangGraph pipeline
    # ------------------------------------------------------------------
    initial_state = {
        "query": safe_query,
        "user_role": request.user_role,
        "retrieval_mode": request.retrieval_mode,
        "answer_mode": request.answer_mode,
        "chat_history": request.chat_history,
        "error": None,
    }

    try:
        # ainvoke: asynchroniczne wykonanie grafu od START do END.
        # LangGraph scala częściowe stany zwrócone przez każdy węzeł.
        # final_state to kompletny RAGState po przejściu całego grafu.
        final_state = await rag_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd pipeline RAG: {exc}",
        )

    # Sprawdź czy węzeł answer_generator ustawił błąd
    if final_state.get("error"):
        # Błąd API (np. rate limit Anthropic) — zwracamy graceful odpowiedź
        # zamiast HTTP 500. Użytkownik dostaje informację o problemie.
        # Odpowiedź jest już ustawiona przez answer_generator na komunikat błędu.
        pass  # kontynuujemy — odpowiedź jest w final_state["answer"]

    # ------------------------------------------------------------------
    # Krok 3: Mapowanie stanu grafu na AskResponse
    # ------------------------------------------------------------------
    tokens = final_state.get(
        "tokens",
        TokenUsage(context=0, answer=0, total=0),
    )
    latency = round((time.monotonic() - start) * 1000, 2)

    return AskResponse(
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        retrieval_mode=request.retrieval_mode,
        tokens=tokens,
        confidence=final_state.get("confidence", "cautious"),
        latency_ms=latency,
    )
