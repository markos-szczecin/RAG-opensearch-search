"""
VectorSearchService — wyszukiwanie semantyczne przez przybliżone k-NN (kNN) w OpenSearch.

Jak działa wyszukiwanie wektorowe?
-------------------------------------
Każdy chunk dokumentu jest zakodowany przez model embeddings (OpenAI text-embedding-3-small)
jako wektor 1536 liczb zmiennoprzecinkowych. Wektor reprezentuje "znaczenie" tekstu
w przestrzeni matematycznej, gdzie teksty o podobnym znaczeniu leżą blisko siebie.

Przy wyszukiwaniu:
  1. Zapytanie użytkownika jest zakodowane tym samym modelem.
  2. OpenSearch szuka k wektorów dokumentów najbliższych zapytaniu.
  3. "Bliskość" mierzona jest cosine similarity (kąt między wektorami).

HNSW (Hierarchical Navigable Small World) — algorytm indeksowania
-------------------------------------------------------------------
HNSW buduje wielowarstwowy graf, gdzie węzły to wektory, a krawędzie łączą
"sąsiadów". Wyszukiwanie nawiguje ten graf hierarchicznie od rzadszych
warstw do gęstszych. Parametry:
  - m=16: liczba krawędzi per węzeł — więcej = lepsza jakość, więcej pamięci
  - ef_construction=128: szerokość przeszukiwania podczas budowy grafu
  - ef_search=512: szerokość przeszukiwania podczas zapytania

Kompromis recall vs latency:
  - ef_search=512 → wysoki recall (~99%), ale ~20ms latency
  - ef_search=64  → niższy recall (~95%), ale ~5ms latency

efficient_filter — kluczowa optymalizacja
------------------------------------------
Problem: bez filtrowania, HNSW przegląda całą przestrzeń wektorową i POTEM
eliminuje wyniki niespełniające filtrów metadanych. Jeśli filtr jest restrykcyjny
(np. tylko dokumenty "confidential"), większość wyników HNSW zostaje odrzucona
i możemy dostać mniej niż k wyników.

Rozwiązanie: efficient_filter powoduje że HNSW bierze filtry pod uwagę PODCZAS
nawigacji grafu, przycinając gałęzie od razu. Wynik: zarówno lepsza wydajność
(mniej wektorów do sprawdzenia) jak i gwarancja k wyników.

Wymagania: efficient_filter działa tylko gdy FilterBuilder zwraca bool filter
(nie match_all). Po naszych zmianach filter ZAWSZE zawiera przynajmniej filtr
roli dostępu, więc efficient_filter jest zawsze możliwy.
"""

import time

from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.indexing.embedder.base import Embedder
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.search.base import SearchService
from app.search.filters import FilterBuilder


class VectorSearchService(SearchService):
    """
    Wyszukiwanie semantyczne przez kNN z embeddings OpenAI.

    Przepływ zapytania:
      1. Embedowanie zapytania (~80-150 ms — dominujące opóźnienie)
      2. Budowa ciała zapytania kNN z efficient_filter
      3. Wykonanie zapytania OpenSearch (~5-20 ms dla indeksu <100K dokumentów)
      4. Parsowanie wyników

    Uwaga: latency embedding API jest zwykle 5-10x większa niż latency OpenSearch.
    Dlatego mierzymy je osobno — to ważna informacja przy optymalizacji systemu.
    Rozwiązanie długoterminowe: cache embeddingów zapytań (LRU na ostatnie N zapytań).
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        embedder: Embedder,
        index_name: str,
        settings: Settings,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._index = index_name
        self._filter_builder = FilterBuilder(settings)
        self._settings = settings

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Wykonuje wyszukiwanie kNN i zwraca ranking semantycznie podobnych chunków.

        Mierzymy latency embeddingu i wyszukiwania OSOBNO dla profilowania.
        Jeśli całkowity czas jest zbyt długi, od razu wiadomo który etap
        jest wąskim gardłem — bez osobnych metryk trzeba by zgadywać.
        """
        start = time.monotonic()

        # Krok 1: zakoduj zapytanie jako wektor
        # To wywołanie sieciowe do OpenAI — dominujące opóźnienie w tym serwisie
        embed_start = time.monotonic()
        query_vector = await self._embedder.embed_query(request.query)
        embed_ms = (time.monotonic() - embed_start) * 1000

        # Krok 2: zbuduj zapytanie kNN
        query_body = self._build_query(request, query_vector)

        # Krok 3: wykonaj wyszukiwanie w OpenSearch
        response = await self._client.search(index=self._index, body=query_body)
        results = self._parse_response(response)

        total = response.get("hits", {}).get("total", {}).get("value", 0)

        latency = (time.monotonic() - start) * 1000
        return SearchResponse(
            results=results,
            total=total,
            retrieval_mode="vector",
            latency_ms=round(latency, 2),
        )

    def _build_query(self, request: SearchRequest, query_vector: list[float]) -> dict:
        """
        Buduje zapytanie kNN z efficient_filter dla pre-filtrowania.

        Dlaczego efficient_filter jest lepszy od post-filtra?
        -------------------------------------------------------
        Post-filtr (zastosowany po wyszukiwaniu HNSW):
          - HNSW przeszukuje CAŁĄ przestrzeń wektorową
          - Wyniki są filtrowane dopiero po znalezieniu top-k kandydatów
          - Problem: jeśli filtr jest restrykcyjny, możemy dostać 0-2 wyniki
            zamiast żądanych 10

        efficient_filter (filtr zintegrowany z HNSW):
          - HNSW uwzględnia filtr PODCZAS nawigacji grafu
          - Gałęzie prowadzące do odfiltrowanych dokumentów są przycinane
          - Gwarancja: zawsze dostajemy k wyników (jeśli tyle spełnia filtr)
          - Dodatkowy benefit: szybsze przeszukiwanie (mniejsza przestrzeń)

        Kiedy efficient_filter jest dostępny?
        ---------------------------------------
        Wymaga non-trivial filter — nie match_all. Po naszych zmianach
        FilterBuilder ZAWSZE generuje bool filter (przynajmniej filtr roli),
        więc efficient_filter jest zawsze aktywny.
        """
        filter_clause = self._filter_builder.build(request.filters, request.user_role)

        knn_params: dict = {
            "vector": query_vector,
            "k": request.top_k,
        }

        # efficient_filter jest dostępny gdy mamy rzeczywisty filtr (bool),
        # nie match_all (co byłoby puste — OpenSearch nie akceptuje match_all
        # jako efficient_filter)
        if "bool" in filter_clause:
            knn_params["efficient_filter"] = filter_clause

        return {
            "size": request.top_k,
            "query": {
                "knn": {
                    "content_vector": knn_params
                }
            },
        }

    def _parse_response(self, response: dict) -> list[SearchResult]:
        """
        Mapuje odpowiedź OpenSearch kNN na listę SearchResult.

        Różnice względem keyword._parse_response:
          - Brak pola "highlight" — kNN nie generuje snippetów tekstowych
          - _score to cosine similarity z zakresu [0, 1] (już znormalizowana)
            Dla keyword _score to BM25 score (zakres zależy od długości dokumentu)
          - Dlatego przed łączeniem wyników w hybrid search potrzebna jest
            osobna normalizacja min-max dla każdego trybu

        Cosine similarity: 1.0 = identyczne wektory, 0.0 = prostopadłe (brak związku)
        W praktyce wyniki rzadko schodzą poniżej 0.3 — większość tekstów ma
        jakiś wspólny słownik, więc ich wektory nie są całkowicie prostopadłe.
        """
        hits = response.get("hits", {}).get("hits", [])
        results = []

        for hit in hits:
            src = hit.get("_source", {})

            results.append(
                SearchResult(
                    chunk_id=src.get("chunk_id", hit["_id"]),
                    doc_id=src.get("doc_id", ""),
                    title=src.get("title", ""),
                    content=src.get("content", ""),
                    # _score z kNN to cosine similarity [0, 1]
                    score=hit.get("_score", 0.0),
                    source_path=src.get("source_path", ""),
                    access_level=src.get("access_level", "public"),
                    doc_type=src.get("doc_type", ""),
                    # kNN nie generuje highlightów — zawsze None
                    highlight=None,
                    status=src.get("status"),
                    version=src.get("version"),
                )
            )

        return results
