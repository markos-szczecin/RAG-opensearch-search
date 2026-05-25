"""
Feedback endpoint — zbiera oceny użytkowników odpowiedzi RAG.

Dlaczego zbieranie feedbacku jest kluczowe?
--------------------------------------------
System RAG działa na danych statycznych (indeks dokumentów) i bez feedbacku
nie "uczy się" z błędów. Feedback pozwala na:

  1. Identyfikację słabych punktów:
     Zapytania z ratingiem ≤ 2 wskazują gdzie retrieval lub generowanie zawodzą.
     Analiza tych przypadków kieruje ulepszeniami (nowe synonimy, zmiana chunk_size).

  2. Ewaluację zmian:
     Gdy zmieniamy parametry (inny model embeddings, inna wielkość chunku),
     porównujemy średni rating przed i po. Bez tej metryki jesteśmy ślepi.

  3. Priorytetyzację re-indeksowania:
     Dokumenty z wieloma niskimi ratingami potrzebują przeglądu — możliwe że
     treść jest nieaktualna lub brakuje kluczowych informacji.

  4. Monitorowanie trendów:
     Nagły spadek średniego ratingu sugeruje problem (zmiana API, zepsute dokumenty).

Powiązanie feedbacku z odpowiedzią:
  Frontend powinien przesyłać dokładny tekst query i answer z sesji
  (nie pozwalając użytkownikowi edytować). To gwarantuje że wiemy dokładnie
  którą wersję odpowiedzi ocenił użytkownik — ważne gdy system jest aktualizowany.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_eval_repository
from app.evaluation.repository import EvalRepository
from app.models.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    repo: EvalRepository = Depends(get_eval_repository),
) -> FeedbackResponse:
    """
    Przyjmuje ocenę (1-5 gwiazdek) i opcjonalny komentarz do odpowiedzi RAG.

    Dane są zapisywane asynchronicznie do PostgreSQL.
    Wysoka latency bazy nie wpływa na UX — odpowiadamy natychmiast po zapisie.

    Args:
        request:  FeedbackRequest z query, answer, rating (1-5), comment.
        repo:     EvalRepository (singleton z DI).

    Returns:
        FeedbackResponse z UUID zapisanego feedbacku.
        Frontend może go użyć do ewentualnego usunięcia lub aktualizacji opinii.

    Raises:
        HTTPException 500: gdy zapis do bazy danych nie powiedzie się.
    """
    try:
        feedback_id = await repo.log_feedback(request)
        return FeedbackResponse(feedback_id=feedback_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd zapisu feedbacku: {exc}",
        )
