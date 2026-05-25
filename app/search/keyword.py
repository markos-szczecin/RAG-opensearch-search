"""
KeywordSearchService — wyszukiwanie leksykalne BM25 przez OpenSearch multi_match.

Czym jest BM25 i dlaczego go używamy?
---------------------------------------
BM25 (Best Match 25, Robertson & Zaragoza 2009) to ewolucja klasycznego TF-IDF.
Kluczowe usprawnienia względem TF-IDF:
  1. Nasycenie częstości terminu (term frequency saturation): po pewnej liczbie
     wystąpień słowa w dokumencie wzrost wyniku wyhamowuje. Zapobiega to
     dominacji dokumentów, które wielokrotnie powtarzają szukane słowo.
  2. Normalizacja długości dokumentu: krótki FAQ z jednym zdaniem o "limicie
     przelewu" uzyskuje porównywalny wynik do długiej polityki, która o tym
     samym wspomina raz pośród 200 akapitów. Bez normalizacji długi dokument
     zdobywałby wysokie wyniki tylko dlatego, że jest długi.

Kiedy BM25 jest lepszy od wyszukiwania wektorowego?
-----------------------------------------------------
  - Dokładne nazwy produktów: "karta Visa Infinite" — model embeddings może
    nie rozróżniać jej od "karta Mastercard Platinum"
  - Numery polis, kody dokumentów: "POL-2024-001-A" — unikalne tokeny, brak
    semantycznego podobieństwa do nauczenia
  - Terminy prawne i regulacyjne: dokładne brzmienie przepisu ma znaczenie
  - Akronimy: "MFA", "IBAN", "KYC" — synonimy są zdefiniowane w analizatorze
    (fintech_analyzer), ale BM25 niezawodnie dopasowuje dokładne wystąpienia

Kiedy BM25 jest gorszy?
------------------------
  - Parafrazowane pytania: "jak zwiększyć limit?" vs "podniesienie pułapu transakcji"
    — różne słowa, to samo znaczenie. Tu wygrywa wyszukiwanie wektorowe.
  - Pytania konceptualne: "czy konto jest bezpieczne?" — wymaga rozumienia znaczenia

Dlatego domyślny tryb to "hybrid" łączący oba podejścia.
"""

import time

from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.search.base import SearchService
from app.search.filters import FilterBuilder


class KeywordSearchService(SearchService):
    """
    Implementacja wyszukiwania BM25 przez OpenSearch multi_match.

    Struktura zapytania (kolejność klauzul ma znaczenie dla wydajności):
      1. must → multi_match: główne kryterium dopasowania tekstu
      2. should → match_phrase: bonus za dokładne wystąpienie frazy
      3. filter → FilterBuilder: metadane (rola, daty, status) — bez wpływu na _score

    Dlaczego highlight jest ważny dla RAG?
    ----------------------------------------
    Highlight zwraca fragment tekstu z zaznaczonym dopasowaniem (np. <em>limit</em>).
    W interfejsie użytkownika pokazujemy ten snippet zamiast całego chunka —
    użytkownik od razu widzi DLACZEGO dany wynik pasuje do zapytania.
    W RAG nie używamy highlightu do generowania odpowiedzi (używamy pełnego contentu),
    ale jest cennym sygnałem dla debugowania jakości wyszukiwania.
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        index_name: str,
        settings: Settings,
    ) -> None:
        self._client = client
        self._index = index_name
        self._filter_builder = FilterBuilder(settings)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Wykonuje zapytanie BM25 i zwraca rankinig wyników.

        Mierzy latency od momentu budowy zapytania do sparsowania odpowiedzi.
        Latency wyszukiwania keyword jest zazwyczaj 5-20 ms dla indeksu <1M dokumentów.
        """
        start = time.monotonic()

        query_body = self._build_query(request)

        response = await self._client.search(index=self._index, body=query_body)
        results = self._parse_response(response)

        # response["hits"]["total"] to dict {"value": N, "relation": "eq"}, nie int.
        # To zmiana wprowadzona w Elasticsearch/OpenSearch 7.x — zawsze używaj ["value"].
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        latency = (time.monotonic() - start) * 1000
        return SearchResponse(
            results=results,
            total=total,
            retrieval_mode="keyword",
            latency_ms=round(latency, 2),
        )

    def _build_query(self, request: SearchRequest) -> dict:
        """
        Buduje kompletny dict zapytania OpenSearch.

        Struktura zapytania bool:
          must   = warunki OBOWIĄZKOWE (niespełnienie → wynik 0)
          should = warunki POŻĄDANE (spełnienie → bonus do _score)
          filter = warunki FILTRUJĄCE (bez wpływu na _score, ale cachowane)

        Dlaczego "best_fields" zamiast "most_fields" w multi_match?
        --------------------------------------------------------------
        "best_fields" bierze najwyższy wynik spośród pól (content, title^2).
        Zapobiega to sztucznemu zawyżaniu score przez sumy z wielu pól,
        gdy dokument ma mało treści ale tytuł idealnie pasuje do zapytania.
        "title^2" oznacza, że dopasowanie w tytule liczy się podwójnie —
        tytuł to skrót całego dokumentu, więc jest silniejszym sygnałem.

        Dlaczego fuzziness="AUTO" tylko na content?
        ----------------------------------------------
        Fuzziness (tolerancja literówek) na tytule spowodowałoby fałszywe
        dopasowania nazw produktów. "Visa Infinite" nie powinno dopasowywać
        "Visa Infinite Plus". Fuzziness na content jest bezpieczniejsza.
        AUTO wybiera odległość edycji 0 dla <3 znaków, 1 dla 3-5, 2 dla >5.

        Dlaczego match_phrase jako should?
        ------------------------------------
        Fraza "limit dzienny przelewu" zawiera te same tokeny co multi_match,
        ale ich kolejność i sąsiedztwo jest ważne. match_phrase boost=1.5
        nagradza dokumenty zawierające DOKŁADNĄ frazę — ważne dla terminologii
        prawnej, nazw produktów i numerów dokumentów.
        minimum_should_match: 0 oznacza że should jest OPCJONALNE —
        brak dopasowania frazy nie wyklucza dokumentu, tylko nie daje bonusu.
        """
        filter_clause = self._filter_builder.build(request.filters, request.user_role)

        return {
            "size": request.top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                # title^2 = boost tytułu o 2x względem contentu
                                "fields": ["content", "title^2"],
                                "type": "best_fields",
                                # Tolerancja na literówki: AUTO = 0 dla <3 znaków,
                                # 1 dla 3-5 znaków, 2 dla >5 znaków
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    # Opcjonalny bonus za dokładne wystąpienie frazy
                    "should": [
                        {
                            "match_phrase": {
                                "content": {
                                    "query": request.query,
                                    "boost": 1.5,
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 0,
                    "filter": filter_clause,
                }
            },
            # Highlight: OpenSearch zwraca fragmenty otoczone tagami <em>...</em>
            # fragment_size=200 to max długość snippetu w znakach
            # number_of_fragments=1 — chcemy jeden najlepszy fragment, nie wiele
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 1,
                    }
                }
            },
        }

    def _parse_response(self, response: dict) -> list[SearchResult]:
        """
        Mapuje surową odpowiedź OpenSearch na listę SearchResult.

        Struktura odpowiedzi OpenSearch (uproszczona):
        {
          "hits": {
            "total": {"value": 42, "relation": "eq"},
            "hits": [
              {
                "_id": "doc-001::chunk-000",
                "_score": 3.14,
                "_source": { "chunk_id": ..., "content": ..., ... },
                "highlight": { "content": ["...fragment z <em>tagami</em>..."] }
              }
            ]
          }
        }

        Highlight jest listą fragmentów — bierzemy pierwszy (najlepszy wg OpenSearch).
        Jeśli highlight nie jest dostępny (np. zapytanie filtrujące bez full-text),
        zwracamy None. UI powinien w takim przypadku pokazać początek contentu.
        """
        hits = response.get("hits", {}).get("hits", [])
        results = []

        for hit in hits:
            src = hit.get("_source", {})

            # Highlight jest opcjonalny — nie każde zapytanie go generuje
            highlight_frags = hit.get("highlight", {}).get("content", [])
            highlight = highlight_frags[0] if highlight_frags else None

            results.append(
                SearchResult(
                    chunk_id=src.get("chunk_id", hit["_id"]),
                    doc_id=src.get("doc_id", ""),
                    title=src.get("title", ""),
                    content=src.get("content", ""),
                    score=hit.get("_score", 0.0),
                    source_path=src.get("source_path", ""),
                    access_level=src.get("access_level", "public"),
                    doc_type=src.get("doc_type", ""),
                    highlight=highlight,
                    # Pola opcjonalne — mapujemy jeśli są w indeksie
                    status=src.get("status"),
                    version=src.get("version"),
                )
            )

        return results
