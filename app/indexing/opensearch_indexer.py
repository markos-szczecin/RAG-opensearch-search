"""
OpenSearchIndexer — zapis chunków dokumentów do indeksu OpenSearch.

Dlaczego upsert zamiast insert?
---------------------------------
Upsert (update-or-insert) oznacza: jeśli dokument z tym _id istnieje — zaktualizuj go;
jeśli nie — utwórz go. To kluczowe dla idempotentności pipeline'u:

  - Przy pierwszym indeksowaniu: wszystkie chunki są nowe → insert
  - Przy re-indeksowaniu po zmianie dokumentu: istniejące chunki są aktualizowane,
    nowe są dodawane, stare (po skróceniu dokumentu) trzeba usunąć osobno przez delete_by_doc_id()
  - Bez upsert: każde uruchomienie tworzyłoby duplikaty → zepsute wyniki wyszukiwania

Dlaczego bulk zamiast indywidualnych zapytań?
----------------------------------------------
Bulk API wysyła wiele operacji w jednym żądaniu HTTP. Porównanie dla 100 chunków:
  - Indywidualnie: 100 żądań HTTP × ~5ms każde = ~500ms + narzut sieci
  - Bulk:           1 żądanie HTTP              = ~20ms + parsing JSON

Mapowanie indeksu (INDEX_MAPPING) definiuje schemat danych:
  - Pola tekstowe (content, title): analizowane przez fintech_analyzer
  - Pole wektorowe (content_vector): knn_vector z HNSW
  - Pola metadanych (chunk_id, doc_id, itp.): keyword — dokładne dopasowanie
  - Pola dat (valid_from, valid_to): date — umożliwia range queries
"""

from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from app.models.document import Chunk, IngestionResult

# ---------------------------------------------------------------------------
# Mapowanie indeksu OpenSearch
# ---------------------------------------------------------------------------
# To jest "schemat tabeli" dla OpenSearch. Jest wersjonowane jako stała
# (zamiast np. ładowania z pliku YAML), bo:
#   1. Zmiany schematu są widoczne w git diff — łatwo śledzić ewolucję
#   2. Błędy składniowe są wykrywane przez linter Pythona, nie dopiero w runtime
#   3. Komentarze wyjaśniają DLACZEGO każde pole ma taki typ
#
# Gdy mapowanie się zmienia (np. nowe pole, zmiana wymiaru wektora):
#   1. Zmień wersję indeksu w IndexManager
#   2. IndexManager automatycznie przeprowadzi re-indeksowanie przez alias
# ---------------------------------------------------------------------------
INDEX_MAPPING: dict = {
    "settings": {
        "index": {
            # knn: True włącza HNSW (Hierarchical Navigable Small World) —
            # algorytm przybliżonego wyszukiwania najbliższych sąsiadów.
            # Bez tej flagi OpenSearch odrzuca knn_vector fields.
            "knn": True,
            # ef_search: szerokość przeszukiwania podczas zapytania.
            # Wyższa wartość = lepszy recall, wyższe opóźnienie.
            # 512 to dobry kompromis dla bazy ~100K dokumentów.
            "knn.algo_param.ef_search": 512,
        },
        "analysis": {
            "analyzer": {
                # fintech_analyzer: niestandardowy analizator dla dokumentów finansowych.
                # Łańcuch przetwarzania: tokenizacja → małe litery → usunięcie stop słów
                #   → stemming → mapowanie synonimów
                "fintech_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop",
                        "english_stemmer",
                        "fintech_synonyms",
                    ],
                }
            },
            "filter": {
                "english_stop": {
                    "type": "stop",
                    # _english_ to wbudowana lista stop słów OpenSearch:
                    # the, is, are, was, were, be, been, being, have, has...
                    # Usunięcie ich zmniejsza szum w indeksie BM25.
                    "stopwords": "_english_",
                },
                "english_stemmer": {
                    "type": "stemmer",
                    # Stemming sprowadza słowa do formy podstawowej:
                    # "transfers" → "transfer", "authenticated" → "authent"
                    # Poprawia recall (mniej pominięć) kosztem lekko niższej precyzji.
                    "language": "english",
                },
                "fintech_synonyms": {
                    "type": "synonym",
                    # Synonimy dziedzinowe — użytkownik szuka "2FA", dokumenty mówią "MFA".
                    # Lista powinna być rozwijana na podstawie realnych zapytań użytkowników.
                    # Format: "term1, term2, term3" → wszystkie mapowane na siebie wzajemnie.
                    "synonyms": [
                        "mfa, 2fa, two-factor authentication",
                        "iban, bank account number",
                    ],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            # ---- Pola przeszukiwalne ----
            "content": {
                "type": "text",
                "analyzer": "fintech_analyzer",
                # Podpole .keyword umożliwia dokładne dopasowanie i agregacje
                # na całym contencie (np. GROUP BY w analityce). Rzadko używane
                # dla długich tekstów — głównie dla tytułów i krótkich pól.
                "fields": {
                    "keyword": {"type": "keyword"},
                },
            },
            "title": {
                "type": "text",
                "analyzer": "fintech_analyzer",
                # boost: 2 — dopasowanie w tytule liczy się podwójnie w BM25.
                # Tytuł to skrót dokumentu — jeśli tytuł pasuje, dokument
                # jest prawdopodobnie bardzo relevantny.
                "boost": 2,
                "fields": {"keyword": {"type": "keyword"}},
            },
            # ---- Pole wektorowe ----
            "content_vector": {
                "type": "knn_vector",
                # dimension musi DOKŁADNIE zgadzać się z wymiarem modelu embeddings.
                # text-embedding-3-small = 1536. Zmiana tutaj wymaga re-indeksowania.
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "parameters": {
                        # ef_construction: szerokość przeszukiwania podczas BUDOWY grafu.
                        # Wyższa = lepszy graf = lepszy recall, ale wolniejsze indeksowanie.
                        "ef_construction": 128,
                        # m: liczba krawędzi per węzeł w grafie HNSW.
                        # 16 to standardowa wartość. Więcej krawędzi = lepszy recall,
                        # ale ~m× więcej pamięci RAM i miejsca na dysku.
                        "m": 16,
                    },
                },
            },
            # ---- Pola identyfikatorów i metadanych ----
            # keyword: dokładne dopasowanie (case-sensitive, bez tokenizacji).
            # Używane dla pól gdzie szukamy KONKRETNEJ wartości, nie tekstu.
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "department": {"type": "keyword"},
            "language": {"type": "keyword"},
            "access_level": {"type": "keyword"},
            "status": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "version": {"type": "integer"},
            # ---- Pola dat ----
            # type: date umożliwia range queries z "now/d", "gte", "lte" itp.
            # Domyślny format: ISO 8601 ("2024-01-15" lub "2024-01-15T00:00:00Z")
            "valid_from": {"type": "date"},
            "valid_to": {"type": "date"},
        }
    },
}


@dataclass
class BulkResult:
    """Wynik operacji bulk upsert — ile chunków zaindeksowano, ile nie."""
    indexed: int
    failed: int
    errors: list[str]


class OpenSearchIndexer:
    """
    Obsługuje operacje zapisu do OpenSearch: upsert i usuwanie chunków.

    Idempotentność: używa doc_as_upsert=True, więc wielokrotne uruchamianie
    pipeline'u na tym samym dokumencie jest bezpieczne — brak duplikatów.

    Jako _id dokumentu OpenSearch używamy chunk_id ("doc-001::chunk-002").
    To czyni każdy chunk unikalny i adresowalny bez dodatkowego wyszukiwania.
    """

    def __init__(self, client: AsyncOpenSearch, index_name: str) -> None:
        self._client = client
        self._index = index_name

    async def bulk_upsert(self, chunks: list[Chunk]) -> BulkResult:
        """
        Zapisuje lub aktualizuje partię chunków w OpenSearch.

        Używa opensearchpy.helpers.async_bulk do wysyłki wszystkich chunków
        w jednym żądaniu HTTP (minimalizuje narzut sieci i czas oczekiwania).

        Dlaczego raise_on_error=False?
        --------------------------------
        Domyślnie async_bulk rzuca wyjątek przy PIERWSZYM błędzie, anulując
        cały batch. To złe zachowanie przy ingestion:
          - Jeden uszkodzony chunk (np. zbyt duży wektor) nie powinien blokować
            499 poprawnych chunków
          - Błędy są logowane i zwracane w BulkResult — caller może zdecydować
            czy to krytyczne (100% błędów) czy ignorowalne (1% błędów)

        doc_as_upsert=True: semantyka upsert — aktualizuj jeśli istnieje,
        utwórz jeśli nie istnieje. Bez tego "update" na nieistniejącym
        dokumencie zwróciłby błąd.

        Args:
            chunks: Lista chunków z wypełnionym content_vector.
                    Chunki bez wektora będą miały wektor zer — to słaby sygnał
                    dla kNN, ale nie spowoduje błędu.

        Returns:
            BulkResult z liczbą zaindeksowanych, nieudanych i listą błędów.
        """
        if not chunks:
            return BulkResult(indexed=0, failed=0, errors=[])

        actions = [
            {
                "_op_type": "update",
                "_index": self._index,
                # chunk_id jako _id dokumentu OpenSearch — gwarantuje unikalność
                "_id": chunk.chunk_id,
                "doc": chunk.to_opensearch_doc,
                # Utwórz dokument jeśli nie istnieje, zaktualizuj jeśli istnieje
                "doc_as_upsert": True,
            }
            for chunk in chunks
        ]

        success, errors = await async_bulk(
            self._client,
            actions,
            raise_on_error=False,
            # chunk_size: ile operacji w jednym żądaniu do OpenSearch.
            # 500 to domyślna wartość opensearchpy — dobry kompromis.
            # Zmniejsz do 100-200 jeśli chunki mają bardzo duże wektory (> 10KB JSON).
            chunk_size=500,
        )

        return BulkResult(
            indexed=success,
            failed=len(errors),
            errors=[str(e) for e in errors],
        )

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Usuwa wszystkie chunki należące do dokumentu.

        Używa delete_by_query zamiast usuwania chunk po chunk, bo:
          1. Nie musimy znać listy chunk_ids — jedno zapytanie usuwa wszystkie
          2. Jest atomiczne z punktu widzenia wyszukiwania — albo wszystkie usunięte,
             albo żadne (w przypadku błędu)
          3. Szybsze: jeden round-trip HTTP zamiast N

        Args:
            doc_id: Identyfikator dokumentu (np. "mobile-auth-policy").

        Returns:
            Liczba usuniętych chunków.
        """
        response = await self._client.delete_by_query(
            index=self._index,
            body={
                "query": {
                    # term filter: dokładne dopasowanie doc_id (keyword field)
                    "term": {"doc_id": doc_id}
                }
            },
            # conflicts="proceed": jeśli jakiś chunk jest właśnie indeksowany
            # (wersja konfliktu), pomiń go zamiast zwracać błąd.
            params={"conflicts": "proceed"},
        )
        return response.get("deleted", 0)
