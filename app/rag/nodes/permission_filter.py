"""
LangGraph node: permission_filter

Filtruje chunki znalezione przez wyszukiwanie tak, żeby do LLM trafiły TYLKO
te, do których użytkownik ma rzeczywiste uprawnienia.

Dlaczego ten węzeł istnieje skoro FilterBuilder już filtruje w zapytaniu?
---------------------------------------------------------------------------
Mamy dwie warstwy filtrowania po stronie dostępu:

  Warstwa 1 — FilterBuilder (w search/filters.py):
    Buduje klauzulę filtrowania OpenSearch i wysyła ją jako część zapytania.
    OpenSearch GWARANTUJE, że zwrócone wyniki spełniają filtr.
    To jest najefektywniejsza warstwa — filtrowanie odbywa się w indeksie.

  Warstwa 2 — permission_filter_node (ten plik):
    Weryfikuje każdy chunk po wyszukiwaniu, przed wysłaniem do LLM.
    To "defence in depth" — dodatkowe zabezpieczenie na wypadek:
      - Błędu konfiguracji FilterBuilder (np. brakujący filtr roli)
      - Błędu wdrożenia (np. FilterBuilder nie jest używany w jakimś ścieżce)
      - Przyszłych zmian w SearchService które pominęłyby FilterBuilder

Zasada "least privilege" w RAG:
  Lepiej odfiltrować za dużo (użytkownik nie dostanie odpowiedzi) niż za mało
  (użytkownik zobaczy poufne informacje, które nie powinny trafić do LLM kontekstu).

Ograniczenie aktualnej implementacji:
  SearchResult nie zawiera pola valid_to (data ważności dokumentu).
  FilterBuilder filtruje po valid_to na poziomie OpenSearch, ale tutaj
  nie możemy sprawdzić świeżości. Rozwiązanie: dodaj valid_to do SearchResult
  i mapuj je w _parse_response() obu serwisów wyszukiwania.
"""

from app.config import get_settings
from app.rag.state import RAGState


async def permission_filter_node(state: RAGState) -> dict:
    """
    Filtruje raw_chunks do tych, do których user ma uprawnienia.

    Algorytm:
      1. Pobierz listę dozwolonych poziomów dostępu dla roli użytkownika.
      2. Dla każdego chunku sprawdź czy jego access_level jest na liście.
      3. Odrzuć chunki z niedozwolonym poziomem dostępu.

    Args (ze state):
      raw_chunks:  Lista SearchResult zwrócona przez węzeł retrieve.
      user_role:   Rola użytkownika (np. "customer", "support_agent").

    Returns (partial state):
      filtered_chunks: Podzbiór raw_chunks z dozwolonymi poziomami dostępu.

    Uwaga o "customer" vs "support_agent":
      customer:       widzi tylko ["public"] — dokumenty FAQ, cenniki
      support_agent:  widzi ["public", "internal"] — dodatkowe instrukcje operacyjne
      compliance:     widzi ["public", "internal", "confidential"] — polityki wewnętrzne
      admin:          widzi wszystkie poziomy — pełny dostęp
    """
    settings = get_settings()
    user_role = state.get("user_role", "customer")
    raw = state.get("raw_chunks", [])

    # Pobierz dozwolone poziomy dostępu dla tej roli.
    # Nieznana rola dostaje tylko ["public"] — fail-safe default.
    # Lepiej odmówić dostępu nieznanej roli niż przyznać za dużo.
    allowed_levels = settings.role_access_levels.get(user_role, ["public"])

    filtered = []
    for chunk in raw:
        # Sprawdzenie dostępu: chunk.access_level musi być na liście dozwolonych.
        # Przykład: chunk.access_level = "internal", allowed_levels = ["public"]
        # → "internal" nie jest w ["public"] → chunk odrzucony
        if chunk.access_level in allowed_levels:
            filtered.append(chunk)
        # W przyszłości: loguj odrzucone chunki z powodem dla /debug/search endpoint
        # (np. "chunk-003: access_level=confidential not in ['public', 'internal']")

    return {"filtered_chunks": filtered}
