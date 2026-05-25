"""
LangGraph node: query_classifier

Klasyfikuje zapytanie użytkownika w jednej z czterech kategorii:
  - retrieval   → przejdź do węzła retrieve (wyszukaj dokumenty)
  - smalltalk   → krótkookutcituj odpowiedź bez wyszukiwania
  - unsafe      → odrzuć zapytanie jako potencjalnie niebezpieczne
  - unclear     → poproś o doprecyzowanie

Dlaczego ten węzeł istnieje?
------------------------------
Bez klasyfikatora KAŻDE zapytanie przechodzi przez pełny pipeline:
wyszukiwanie + rerankowanie + generowanie odpowiedzi. To kosztuje ~1-2s i
~2000 tokenów nawet dla "Cześć, jak się masz?" (smalltalk) lub
"Ignore previous instructions" (injection).

Klasyfikator zaoszczędza:
  - ~80-90% kosztu dla smalltalk (brak wyszukiwania i generowania)
  - 100% kosztu dla unsafe (natychmiastowe odrzucenie)
  - Unika "skażenia" kontekstu LLM przez prompt injection

Dlaczego claude-haiku zamiast claude-sonnet?
----------------------------------------------
Klasyfikacja wymaga tylko rozpoznania intencji (jedna z 4 kategorii).
claude-haiku-4-5 jest 10x tańszy i 3x szybszy niż sonnet, a dokładność
dla tej prostej klasyfikacji jest porównywalna.

Zasada: dobieraj model do złożoności zadania. Użycie sonnet do klasyfikacji
to jak używanie Excela do dodawania 2+2 — działa, ale przepłacasz.

Dwuetapowe podejście:
  1. Regex pre-check (< 1ms, bez API): blokuje oczywiste injection patterns
  2. LLM klasyfikacja (50-100ms): dla zapytań które przeszły pre-check
"""

import anthropic

from app.config import get_settings
from app.guardrails.input import _INJECTION_PATTERNS
from app.rag.prompts import QUERY_CLASSIFIER_PROMPT
from app.rag.state import RAGState

# Module-level singleton dla klienta Anthropic.
# Dlaczego singleton zamiast tworzenia w każdym wywołaniu?
#   - Klient HTTP ma connection pool — nie chcemy go tworzyć przy każdym zapytaniu
#   - Inicjalizacja klienta zajmuje ~10ms (certyfikaty TLS, etc.)
#   - W LangGraph node nie mamy wygodnego miejsca na DI przez FastAPI Depends()
_haiku_client: anthropic.AsyncAnthropic | None = None


def _get_haiku_client() -> anthropic.AsyncAnthropic:
    """Zwraca singleton klienta Anthropic Haiku (lazy initialization)."""
    global _haiku_client
    if _haiku_client is None:
        _haiku_client = anthropic.AsyncAnthropic(
            api_key=get_settings().anthropic_api_key
        )
    return _haiku_client


async def query_classifier_node(state: RAGState) -> dict:
    """
    Klasyfikuje zapytanie i ustawia state["query_class"].

    Zwraca partial RAGState dict — LangGraph scala go z istniejącym stanem.

    Etap 1 — Regex pre-check:
    Przed wysłaniem zapytania do LLM sprawdzamy wzorce injection z InputGuardrail.
    Jeśli zapytanie zawiera "ignore previous instructions" lub podobne,
    oznaczamy jako "unsafe" natychmiast. Nie płacimy za LLM call dla oczywistych
    prób ataku.

    Etap 2 — LLM klasyfikacja:
    Haiku otrzymuje zapytanie i zwraca jedno słowo. max_tokens=5 minimalizuje koszt
    i latency — odpowiedź to zawsze 1-2 tokeny ("retrieval" to 3 tokeny w BPE).
    Przy nieoczekiwanej odpowiedzi (np. "I'm not sure") defaultujemy do "retrieval"
    — bezpieczniejszy fallback niż "unsafe" (nie blokujemy legalnych zapytań).
    """
    query = state["query"]

    # ------------------------------------------------------------------
    # Etap 1: Szybki regex pre-check (< 1ms)
    # ------------------------------------------------------------------
    # _INJECTION_PATTERNS to lista skompilowanych wyrażeń regularnych z InputGuardrail.
    # Sprawdzamy je PRZED wywołaniem LLM bo:
    #   a) Oszczędzamy ~100ms i koszt API dla oczywistych ataków
    #   b) Nie chcemy żeby LLM w ogóle "widział" prompt injection — model mógłby
    #      mimo wszystko przetworzyć część instrukcji zanim je odrzuci
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return {"query_class": "unsafe"}

    # ------------------------------------------------------------------
    # Etap 2: Klasyfikacja LLM z claude-haiku-4-5
    # ------------------------------------------------------------------
    try:
        client = _get_haiku_client()
        response = await client.messages.create(
            model="claude-haiku-4-5",
            # max_tokens=5: odpowiedź to jedno słowo (1-3 tokeny)
            # Ustawiamy mały limit by zminimalizować koszt i latency.
            # Jeśli model chciałby odpowiedzieć dłużej, zostanie przerwany.
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": QUERY_CLASSIFIER_PROMPT.format(query=query),
                }
            ],
        )

        raw_class = response.content[0].text.strip().lower()

        # Walidacja odpowiedzi: akceptujemy tylko znane kategorie.
        # Modele językowe czasem zwracają nieoczekiwane odpowiedzi mimo
        # instrukcji "respond with a single word". Default to "retrieval"
        # jest bezpieczny — w najgorszym razie nie-retrieval query
        # dostanie wynik wyszukiwania (który będzie pusty lub nierelwantny).
        valid_classes = {"retrieval", "smalltalk", "unsafe", "unclear"}
        query_class = raw_class if raw_class in valid_classes else "retrieval"

    except anthropic.APIError:
        # Błąd API (timeout, rate limit, etc.): defaultuj do retrieval.
        # Lepiej dać użytkownikowi potencjalnie mniej precyzyjną odpowiedź
        # niż całkowicie odmówić obsługi z powodu błędu klasyfikatora.
        query_class = "retrieval"

    return {"query_class": query_class}


def route_query(state: RAGState) -> str:
    """
    Funkcja routingu dla conditional edges w LangGraph.

    LangGraph wywołuje tę funkcję po zakończeniu query_classifier_node
    i na podstawie zwróconej wartości wybiera kolejny węzeł do wykonania.

    Mapa routingu (zdefiniowana w graph.py):
      "retrieval" → węzeł "retrieve"
      "smalltalk" → węzeł "safe_direct_answer"
      "unsafe"    → węzeł "refusal"
      "unclear"   → węzeł "clarification"
    """
    return state.get("query_class", "retrieval")
