"""
FilterBuilder — zamienia metadane żądania i rolę użytkownika na klauzulę filtrowania OpenSearch.

Dlaczego istnieje ten plik?
----------------------------
Logika filtrowania jest wspólna dla wszystkich trzech trybów wyszukiwania (keyword, vector, hybrid).
Wydzielenie jej do osobnej klasy oznacza, że dodanie nowego pola metadanych (np. "region")
wymaga zmiany TYLKO tutaj — nie trzeba modyfikować każdej usługi wyszukiwania osobno.
To klasyczny wzorzec separacji odpowiedzialności (Single Responsibility Principle).

Jak działają filtry w OpenSearch?
-----------------------------------
OpenSearch rozróżnia dwa konteksty zapytań:
  - "query context": oblicza wynik trafności (_score); używane przez BM25 i kNN
  - "filter context": nie wpływa na _score, ale wykluczającydokumenty; wyniki są cachowane przez OS

Filtry metadanych ZAWSZE idą do filter context — nie mają wpływu na ranking,
ale są szybkie i eliminują dokumenty przed obliczaniem podobieństwa.

Hierarchia ról dostępu w systemie:
------------------------------------
  customer          → public
  support_agent     → public, internal
  compliance        → public, internal, confidential
  admin             → public, internal, confidential, restricted

Ta mapa jest przechowywana w Settings.role_access_levels (z pliku .env),
a nie na stałe w kodzie — dzięki temu można ją zmienić bez przebudowywania obrazu Docker.
"""

from app.config import Settings
from app.models.search import SearchFilters


class FilterBuilder:
    """
    Buduje klauzulę ``bool.filter`` OpenSearch na podstawie filtrów żądania i roli użytkownika.

    Każde wywołanie ``build()`` zwraca gotowy dict gotowy do wklejenia do ciała zapytania OpenSearch.

    Filtry są łączone semantyką AND (wszystkie muszą być spełnione). Kolejność:
      1. Rola użytkownika → dozwolone poziomy dostępu  (ZAWSZE stosowany — fundament bezpieczeństwa)
      2. Aktualność dokumentu: valid_from ≤ dziś                (ZAWSZE stosowany)
      3. Aktualność dokumentu: valid_to ≥ dziś LUB null         (ZAWSZE stosowany)
      4. Jawne nadpisania z SearchFilters (status, język, itp.)  (opcjonalne — tylko gdy podane)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, filters: SearchFilters, user_role: str) -> dict:
        """
        Zwraca klauzulę filtrowania OpenSearch jako dict Pythona.

        Args:
            filters:   Opcjonalne filtry metadanych z żądania (status, język, itp.).
            user_role: Rola użytkownika — determinuje dozwolone poziomy dostępu.
                       Nieznana rola domyślnie otrzymuje tylko dostęp "public".

        Returns:
            Dict OpenSearch gotowy do użycia jako ``query.bool.filter`` lub
            ``knn.content_vector.efficient_filter``.

        Przykładowy wynik dla roli "support_agent":
        {
          "bool": {
            "filter": [
              {"terms": {"access_level": ["public", "internal"]}},
              {"range": {"valid_from": {"lte": "now/d"}}},
              {"bool": {"should": [...], "minimum_should_match": 1}},
              {"term": {"status": "approved"}}
            ]
          }
        }
        """
        must_filters: list[dict] = []

        # ------------------------------------------------------------------ #
        # Filtr 1: Poziom dostępu oparty na roli                              #
        # ------------------------------------------------------------------ #
        # Pobieramy listę dozwolonych poziomów dla tej roli z konfiguracji.
        # Użycie "terms" (liczba mnoga) zamiast "term" jest kluczowe — użytkownik
        # może mieć WIELE dozwolonych poziomów (np. ["public", "internal"]).
        # Domyślnie fallback do ["public"] gdy rola jest nieznana.
        allowed_levels = self._settings.role_access_levels.get(user_role, ["public"])
        must_filters.append({"terms": {"access_level": allowed_levels}})

        # ------------------------------------------------------------------ #
        # Filtr 2: valid_from ≤ dziś (dokument już opublikowany)              #
        # ------------------------------------------------------------------ #
        # Dlaczego "now/d" zamiast str(date.today())?
        #   - "now/d" jest obliczane PO STRONIE SERWERA OpenSearch, nie klienta.
        #     Eliminuje to problemy z różnicą stref czasowych między serwerem
        #     API a serwerem OpenSearch.
        #   - OpenSearch cachuje wyniki filtrów — stały token "now/d" zapewnia,
        #     że cache jest unieważniany raz dziennie, a nie przy każdym zapytaniu.
        #   - "/d" zaokrągla do granicy dnia — nie godziny ani minuty.
        must_filters.append({"range": {"valid_from": {"lte": "now/d"}}})

        # ------------------------------------------------------------------ #
        # Filtr 3: valid_to ≥ dziś LUB pole nie istnieje (dokument bez daty   #
        # wygaśnięcia jest zawsze aktualny)                                    #
        # ------------------------------------------------------------------ #
        # Semantyka: dokument jest aktualny jeśli ALBO nie ma valid_to (null =
        # nigdy nie wygasa), ALBO valid_to jest w przyszłości lub dzisiaj.
        #
        # Realizacja w OpenSearch: bool.should z minimum_should_match: 1
        # oznacza "przynajmniej jeden z poniższych warunków musi być spełniony".
        # To odpowiednik SQL: "valid_to IS NULL OR valid_to >= TODAY"
        must_filters.append({
            "bool": {
                "should": [
                    # Warunek A: pole valid_to w ogóle nie istnieje w dokumencie
                    {"bool": {"must_not": {"exists": {"field": "valid_to"}}}},
                    # Warunek B: valid_to jest dziś lub w przyszłości
                    {"range": {"valid_to": {"gte": "now/d"}}},
                ],
                "minimum_should_match": 1,
            }
        })

        # ------------------------------------------------------------------ #
        # Filtry 4+: jawne nadpisania z SearchFilters                         #
        # ------------------------------------------------------------------ #
        # Te filtry są opcjonalne — stosowane tylko gdy użytkownik je poda.
        # Przykłady użycia:
        #   - status="draft" → administrator sprawdza dokumenty w trakcie review
        #   - language="pl"  → użytkownik chce tylko polskie dokumenty
        #   - department="compliance" → ograniczenie do działu zgodności
        #
        # UWAGA: access_level z SearchFilters nakłada się na filtr roli powyżej.
        # Jeśli użytkownik poda access_level="confidential" a jego rola daje
        # tylko ["public"], dostanie pusty wynik (oba filtry muszą być spełnione).
        if filters.status:
            must_filters.append({"term": {"status": filters.status}})
        if filters.language:
            must_filters.append({"term": {"language": filters.language}})
        if filters.doc_type:
            must_filters.append({"term": {"doc_type": filters.doc_type}})
        if filters.department:
            must_filters.append({"term": {"department": filters.department}})
        if filters.access_level:
            # Jawne nadpisanie poziomu dostępu — używane np. przez panel admina
            # do filtrowania dokumentów konkretnego poziomu
            must_filters.append({"term": {"access_level": filters.access_level}})

        # Jeśli nie ma żadnych filtrów (edge case przy braku roli i pustych filtrach),
        # zwróć match_all zamiast pustego bool — to poprawna składnia OpenSearch.
        return {"bool": {"filter": must_filters}} if must_filters else {"match_all": {}}
