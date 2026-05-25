"""
Retrieval-layer guardrail — filtrowanie chunków po wyszukiwaniu.

Ten guardrail uruchamia się PO wyszukiwaniu, ale ZANIM LLM zobaczy treść dokumentów.
Zapewnia że do promptu trafiają tylko autoryzowane, aktualne i odpowiednie fragmenty.

Dlaczego guardrail zamiast tylko filtrów w zapytaniu?
------------------------------------------------------
FilterBuilder buduje filtry dla OpenSearch i jest bardzo efektywny.
RetrievalGuardrail to dodatkowa warstwa weryfikacji po stronie aplikacji — "belts and suspenders":
  - Odporna na błędy konfiguracji OpenSearch
  - Może implementować bardziej złożone reguły (np. role-based content redaction)
  - Daje audytowalność — każde odrzucenie jest logowalne z powodem

Hierarchia poziomów dostępu:
  public < internal < confidential < restricted
  Wyższa liczba = więcej uprawnień = więcej widocznych dokumentów.

Uwaga na temat świeżości dokumentów:
  Pole valid_to nie jest mapowane do SearchResult w aktualnej implementacji.
  Filtry dat (valid_from ≤ now, valid_to ≥ now) są stosowane przez FilterBuilder
  na poziomie OpenSearch. Metoda _is_allowed() sprawdza tylko access_level.
  Dodaj valid_to do SearchResult i rozszerz _is_allowed() o sprawdzenie daty
  jako ulepszenie Phase 3.
"""

from typing import Any

from app.config import get_settings
from app.guardrails.base import Guardrail, GuardrailResult
from app.models.search import SearchResult

# Hierarchia dostępu: wyższy indeks = wyższy poziom poufności
# Używana do logowania (nie do decyzji dostępu — ta jest przez in-list check)
_ACCESS_HIERARCHY = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


class RetrievalGuardrail(Guardrail):
    """
    Sprawdza czy chunki zwrócone z wyszukiwania są dostępne dla danego użytkownika.

    Używany przez permission_filter_node (jako alternatywna implementacja)
    i bezpośrednio w debug endpoint (/debug/search) do raportowania odrzuceń.
    """

    @property
    def name(self) -> str:
        return "retrieval_guardrail"

    def check(self, payload: Any) -> GuardrailResult:
        """
        Sprawdza całą listę chunków i raportuje ile zostałoby odrzuconych.

        payload: tuple[list[SearchResult], str]  →  (chunks, user_role)

        Używane przez debug endpoint do statystyk, nie do faktycznego filtrowania.
        Do filtrowania używaj filter_chunks().
        """
        chunks, user_role = payload
        settings = get_settings()
        allowed_levels = settings.role_access_levels.get(user_role, ["public"])

        rejected = [c for c in chunks if not self._is_allowed(c, allowed_levels)]
        if rejected:
            return GuardrailResult(
                passed=False,
                action="warn",
                reason=f"{len(rejected)} chunk(s) removed by access/freshness rules",
            )
        return GuardrailResult(passed=True, action="allow")

    def filter_chunks(
        self, chunks: list[SearchResult], user_role: str
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """
        Filtruje chunki i zwraca (dozwolone, odrzucone).

        Zwracanie osobnej listy odrzuconych jest przydatne dla debug endpoint —
        możemy pokazać użytkownikowi (adminowi) które chunki zostały zablokowane
        i dlaczego, co pomaga w debugowaniu konfiguracji ról dostępu.

        Args:
            chunks:    Lista SearchResult do weryfikacji.
            user_role: Rola użytkownika determinująca allowed_levels.

        Returns:
            Tuple (allowed_chunks, rejected_chunks).
        """
        settings = get_settings()
        allowed_levels = settings.role_access_levels.get(user_role, ["public"])
        allowed, rejected = [], []
        for chunk in chunks:
            (allowed if self._is_allowed(chunk, allowed_levels) else rejected).append(chunk)
        return allowed, rejected

    def _is_allowed(self, chunk: SearchResult, allowed_levels: list[str]) -> bool:
        """
        Sprawdza czy pojedynczy chunk jest dostępny dla użytkownika.

        Kryteria:
          1. access_level chunku musi być na liście allowed_levels dla roli.
             Przykład: chunk.access_level = "internal",
                       allowed_levels = ["public"] → odrzucony
                       allowed_levels = ["public", "internal"] → dozwolony

        W przyszłości (Phase 3) można dodać:
          2. Status: chunk.status != "draft" (szkice tylko dla adminów)
          3. Świeżość: chunk.valid_to is None or chunk.valid_to >= date.today()
             (wymaga dodania valid_to do SearchResult)

        Args:
            chunk:          SearchResult do sprawdzenia.
            allowed_levels: Lista dozwolonych poziomów dostępu dla roli.

        Returns:
            True jeśli chunk jest dostępny, False jeśli powinien być odrzucony.
        """
        return chunk.access_level in allowed_levels
