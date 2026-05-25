"""
OpenAIEmbedder — koduje teksty jako wektory liczbowe przez API OpenAI.

Co to jest embedding i do czego służy?
----------------------------------------
Embedding to reprezentacja tekstu jako listy liczb (wektora). Model embeddings
(text-embedding-3-small) przetwarza tekst i zwraca wektor 1536 liczb.

Teksty o podobnym znaczeniu mają podobne wektory (mała odległość kątowa).
To pozwala na wyszukiwanie semantyczne: "limit przelewu" i "maksymalna kwota
transakcji" dadzą podobne wektory, choć nie mają wspólnych słów.

Dlaczego text-embedding-3-small, a nie large?
----------------------------------------------
  - small: 1536 wymiarów, 62K tokenów/USD, ~0.5ms/token
  - large: 3072 wymiarów, 9K tokenów/USD, ~1ms/token
  - Dla dokumentów fintech, small daje porównywalną jakość przy ~7x niższym koszcie.
  - large warto rozważyć gdy mamy bardzo techniczną terminologię w wielu językach.

Trzy warstwy ochrony przed problemami produkcyjnymi
----------------------------------------------------
1. Cache SHA-256: unika ponownego embeddingu identycznych tekstów (re-indexing)
2. Batching: wysyła do 100 tekstów w jednym zapytaniu API zamiast N osobnych
3. Retry z exponential back-off: radzi sobie z chwilowymi błędami rate limit

Bez tych mechanizmów:
  - Re-indexing 1000 dokumentów kosztowałby tyle co pierwsze indeksowanie
  - Jeden błąd 429 (rate limit) przerywałby całą ingestion pipeline
  - 1000 pojedynczych zapytań zamiast 10 batch = 100x więcej narzutu sieci
"""

import asyncio
import hashlib

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.indexing.embedder.base import Embedder

# OpenAI zaleca batch ≤ 2048 stringów na jedno zapytanie.
# 100 to bezpieczna wartość dla długich dokumentów fintech (akapity mogą mieć
# nawet kilkaset tokenów). Zmniejsz do 50 jeśli widzisz błędy "request too large".
_BATCH_SIZE = 100


class OpenAIEmbedder(Embedder):
    """
    Implementacja embeddings przez OpenAI text-embedding-3-small (domyślnie).

    Cache działa na poziomie procesu (in-memory dict). To jest wystarczające
    dla MVP — survives re-indexing w tej samej sesji Pythona. W środowisku
    produkcyjnym rozważ Redis jako cache współdzielony między instancjami.

    Wymiar embeddings jest konfigurowany z Settings.embedding_dimensions (1536).
    Jeśli zmienisz model na large (3072 wymiarów), musisz też zaktualizować
    INDEX_MAPPING w opensearch_indexer.py (knn_vector.dimension).
    """

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        # Słownik {sha256_hash: wektor} — klucz to hash TREŚCI, nie doc_id.
        # Dzięki temu dwa dokumenty z identycznym fragmentem tekstu dzielą
        # ten sam wektor — ważne przy wersjach dokumentów z małymi zmianami.
        self._cache: dict[str, list[float]] = {}

    @property
    def dimensions(self) -> int:
        """Liczba wymiarów wektora — musi zgadzać się z INDEX_MAPPING."""
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Koduje listę tekstów jako wektory, z cache'owaniem, batchingiem i retry.

        Algorytm:
          1. Przeskanuj listę: dla każdego tekstu sprawdź cache.
          2. Niezakeszowane teksty zebrane w jedną listę + mapowanie indeksów.
          3. Wywołaj API w grupach _BATCH_SIZE z retry przy błędach.
          4. Uzupełnij wyniki z cache + API w oryginalnej kolejności.

        Zachowanie kolejności jest krytyczne — wyniki muszą odpowiadać
        dokładnie tekstom na tych samych pozycjach wejściowej listy.
        Dlatego używamy osobnej listy wyników i mapowania indeksów.

        Args:
            texts: Lista tekstów do zakodowania. Może być pusta.

        Returns:
            Lista wektorów w tej samej kolejności co wejściowa lista texts.
        """
        if not texts:
            return []

        # Warstwa 1: Cache lookup
        # Inicjalizujemy wyniki jako puste listy — zastąpimy je wektorami poniżej.
        results: list[list[float]] = [[] for _ in texts]
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            # Wszystkie teksty w cache — zero wywołań API
            return results

        # Warstwa 2: Batching
        # Dzielimy uncached_texts na grupy po _BATCH_SIZE i wysyłamy
        # osobne zapytania API dla każdej grupy.
        for batch_start in range(0, len(uncached_texts), _BATCH_SIZE):
            batch = uncached_texts[batch_start : batch_start + _BATCH_SIZE]

            # Warstwa 3: Retry (dekorator na _embed_with_retry)
            api_response = await self._embed_with_retry(batch)

            for j, embedding_obj in enumerate(api_response.data):
                vec = embedding_obj.embedding
                original_idx = uncached_indices[batch_start + j]
                original_text = uncached_texts[batch_start + j]

                # Zapisz w cache na przyszłe wywołania
                key = self._cache_key(original_text)
                self._cache[key] = vec

                # Wstaw w odpowiednie miejsce wynikowej listy
                results[original_idx] = vec

        return results

    async def embed_query(self, text: str) -> list[float]:
        """
        Koduje pojedyncze zapytanie wyszukiwania.

        Dla modeli text-embedding-3-* instrukcje dla zapytania i dokumentu są
        identyczne (nie ma osobnych input_type="query" / "document").
        Modele E5 i Instructor wymagają osobnych prefiksów — pamiętaj o tym
        przy zmianie modelu embeddings.
        """
        results = await self.embed_texts([text])
        return results[0]

    @retry(
        # Czekaj wykładniczo: 2s → 4s → 8s między próbami
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Maksymalnie 3 próby — po 3 błędach rzuć wyjątek dalej
        stop=stop_after_attempt(3),
        # Retry TYLKO na znane błędy przejściowe:
        #   - RateLimitError (429): przekroczony limit zapytań/minutę
        #   - APIConnectionError: chwilowy problem z siecią
        # Inne błędy (AuthenticationError, InvalidRequestError) nie są
        # przejściowe — retry nie pomoże, lepiej od razu zgłosić błąd.
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError)
        ),
    )
    async def _embed_with_retry(self, texts: list[str]):
        """
        Wywołuje OpenAI Embeddings API z automatycznym retry.

        Dekorator @retry z tenacity obsługuje logikę powtarzania —
        nie ma tu pętli ręcznej. To czytelniejsze i łatwiejsze do testowania.

        Zwraca surową odpowiedź API — lista embeddings w response.data.
        """
        return await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

    @staticmethod
    def _cache_key(text: str) -> str:
        """
        Generuje klucz cache jako SHA-256 hash treści.

        Dlaczego SHA-256 zamiast prostszego hash()?
        ----------------------------------------------
        hash() Pythona:
          - Zmienia się między uruchomieniami programu (PYTHONHASHSEED)
          - Jest podatny na kolizje dla długich stringów
          - Nie jest deterministyczny między procesami

        SHA-256:
          - Deterministyczny (ten sam tekst → ten sam hash zawsze)
          - Odporny na kolizje (praktycznie zerowe ryzyko dwóch różnych
            tekstów z tym samym hashem)
          - Standardowy — można go przetransportować do Redis lub innego
            zewnętrznego cache bez problemu

        W produkcji klucz cache powinien zawierać też wersję modelu:
        f"{model}:{sha256}" — zmiana modelu embeddings wymusza re-embedding.
        """
        return hashlib.sha256(text.encode()).hexdigest()
