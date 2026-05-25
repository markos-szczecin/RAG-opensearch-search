"""
ContextualCompressor — wyodrębnia relevantne zdania z chunków dokumentów.

Problem do rozwiązania
------------------------
Chunki z wyszukiwania mają zwykle 300-500 tokenów. Większość z nich to kontekst
otoczenia — tylko 50-100 tokenów bezpośrednio odpowiada na pytanie.

Wysyłając pełny chunk do LLM:
  - Tracimy budżet tokenów na nieistotny tekst
  - Możemy zmieścić 5 chunków zamiast 10 w tym samym budżecie
  - Model "traci się" w nieistotnym kontekście (bad signal-to-noise ratio)

Strategia kompresji: wyodrębnij TYLKO zdania pasujące do zapytania.

Trzy strategie (od prostej do dokładnej):
  1. Keyword overlap (ta implementacja): scorer = n-gram overlap z zapytaniem.
     Zalety: zero dependencji, < 1ms, deterministyczna.
     Wady: nie rozumie parafraz ("limit" vs "pułap", "transfer" vs "przelew").

  2. Cross-encoder (sentence-transformers): model 90MB uruchamiany lokalnie.
     Zalety: rozumie parafrazy, brak kosztów API.
     Wady: +50-100ms latency, wymaga załadowania modelu do RAM.

  3. LLM extraction (claude-haiku): pytamy model o relevantne zdania.
     Zalety: najdokładniejszy, rozumie kontekst i intencję.
     Wady: +100-200ms latency, koszt API per chunk.

Rekomendacja: zacznij od strategii 1. Gdy masz dane ewaluacyjne, zmierz
precyzję i recall. Przejdź do strategii 2 jeśli jakość jest niewystarczająca.
Strategia 3 jest uzasadniona tylko dla bardzo wymagających aplikacji.

Bezpieczny fallback:
  Jeśli żadne zdanie nie spełnia progu trafności (0.15), kompressor zwraca
  oryginalne chunki bez zmian. To jest kluczowe — nigdy nie usuwamy chunka
  całkowicie tylko dlatego że nie znaleźliśmy pasujących zdań.
"""

import re

from app.models.search import SearchResult

# Słowa pomijane przy porównywaniu overlap z zapytaniem.
# To prosta lista "stop words" — wspólny słownik języka naturalnego.
# Bez filtrowania "the", "is", "of" fałszowanie wyniki: prawie każde zdanie
# zawiera te słowa i dostałoby wysoki overlap.
# Używamy hardcoded setu zamiast biblioteki NLP by uniknąć zależności.
_STOPWORDS = frozenset({
    "what", "how", "when", "where", "who", "which", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "the", "a", "an",
    "to", "for", "of", "in", "on", "at", "by", "with", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once", "i",
    "it", "this", "that", "these", "those", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "not", "only", "own", "same", "than", "too",
    "very", "can", "its", "my", "our", "your", "their", "his", "her",
})

# Minimalny score overlap (0.0 - 1.0) żeby zdanie zostało zakwalifikowane.
# 0.15 oznacza: przynajmniej 15% słów kluczowych zapytania musi być w zdaniu.
# Przykład: zapytanie ma 5 kluczowych słów → zdanie musi zawierać przynajmniej 1.
# Możesz podwyższyć do 0.25-0.3 dla bardziej restrykcyjnej selekcji.
_RELEVANCE_THRESHOLD = 0.15


class ContextualCompressor:
    """
    Skraca chunki do zdań relevantnych dla zapytania.

    Metoda compress() zwraca NOWE obiekty SearchResult z skróconym contentem.
    Oryginalne obiekty pozostają niezmienione — Pydantic model_copy zapewnia
    immutability.

    Score z wyszukiwania jest zachowany bez zmian — kompresja nie wpływa
    na ranking (rerankowanie już się odbyło w poprzednim węźle).
    """

    def __init__(self, strategy: str = "keyword") -> None:
        # Strategia kompresji — tylko "keyword" jest zaimplementowana.
        # Inne strategie mogą być dodane w przyszłości bez zmiany interface'u.
        self._strategy = strategy

    async def compress(
        self,
        chunks: list[SearchResult],
        query: str,
        max_tokens_per_chunk: int = 200,
    ) -> list[SearchResult]:
        """
        Kompresuje każdy chunk wyodrębniając relevantne zdania.

        Dla każdego chunku:
          1. Wyodrębnij słowa kluczowe z zapytania
          2. Podziel chunk na zdania
          3. Oblicz overlap każdego zdania z zapytaniem
          4. Wybierz zdania powyżej progu (lub najlepsze zdanie jako fallback)
          5. Złącz wybranych zdań, przytnij do max_tokens_per_chunk

        Args:
            chunks:              Lista chunków do kompresji.
            query:               Zapytanie użytkownika (dla scoring overlap).
            max_tokens_per_chunk: Maksymalna długość skróconego chunku w tokenach.
                                  Używamy znaku × 4 jako przybliżenia tokenów.

        Returns:
            Lista nowych SearchResult z potencjalnie skróconym contentem.
            Jeśli strategia nie jest "keyword", zwraca chunki bez zmian.
        """
        if self._strategy != "keyword":
            # Inne strategie nie są zaimplementowane — passthrough
            return chunks

        if not chunks:
            return []

        query_terms = self._extract_terms(query)
        if not query_terms:
            # Zapytanie tylko ze stop words (mało prawdopodobne, ale bezpieczny fallback)
            return chunks

        result = []
        for chunk in chunks:
            compressed_content = self._extract_relevant_sentences(
                chunk.content, query_terms, max_tokens_per_chunk
            )
            if compressed_content and compressed_content != chunk.content:
                # Utwórz nowy SearchResult z skróconym contentem
                # model_copy: Pydantic v2 — bezpieczna kopia z nadpisaniem pola
                result.append(chunk.model_copy(update={"content": compressed_content}))
            else:
                # Fallback: zachowaj oryginalny chunk
                # (brak dopasowania, brak zmian lub pusty wynik kompresji)
                result.append(chunk)

        return result

    def _extract_terms(self, text: str) -> set[str]:
        """
        Wyodrębnia istotne słowa kluczowe z tekstu (zapytania lub zdania).

        Algorytm:
          1. Tokenizuj: znajdź wszystkie słowa ≥ 3 znaki (pomijaj cyfry i znaki spec.)
          2. Normalizuj: małe litery
          3. Filtruj: usuń stop words z _STOPWORDS

        Dlaczego minimum 3 znaki?
          Słowa 1-2 znakowe to zwykle przyimki lub skróty (np. "do", "na", "UK").
          Przy tak małej długości stop word lista jest ważniejsza niż cutoff,
          ale 3 znaki to dodatkowy filtr bezpieczeństwa.

        Returns:
            Set unikalnych słów kluczowych (lowercase, bez stop words).
        """
        tokens = re.findall(r"\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}\b", text.lower())
        return {t for t in tokens if t not in _STOPWORDS}

    def _extract_relevant_sentences(
        self,
        content: str,
        query_terms: set[str],
        max_tokens: int,
    ) -> str:
        """
        Wybiera zdania z contentu na podstawie overlap z query_terms.

        Tokenizacja zdań:
          Używamy prostego regex zamiast NLTK/spaCy żeby uniknąć zależności.
          Dzielimy na "." lub "!" lub "?" gdy po nich jest spacja lub newline,
          plus na podwójnych newline'ach (granicach akapitów).
          Niedoskonały, ale wystarczający dla angielskich dokumentów fintech.

        Scoring zdania:
          score = liczba_dopasowań / liczba_terminów_w_zapytaniu
          score ∈ [0.0, 1.0+] — może przekroczyć 1.0 jeśli zdanie zawiera
          więcej kluczowych słów niż zapytanie (zdanie jest bardziej bogate).

        Fallback:
          Jeśli żadne zdanie nie przekracza _RELEVANCE_THRESHOLD = 0.15,
          zwróć najlepiej oceniane zdanie. Nie chcemy zwracać pustego
          chunku — to mogłoby spowodować że retrieval nic nie znajdzie.

        Args:
            content:     Pełna treść chunku.
            query_terms: Set kluczowych słów z zapytania.
            max_tokens:  Maksymalna długość wyniku (przybliżona przez max_tokens * 4 znaki).

        Returns:
            String z wybranymi zdaniami, skrócony do max_tokens * 4 znaków.
            Oryginalna kolejność zdań jest zachowana (nie sortujemy po score).
        """
        # Tokenizacja zdań: podziel na granicach zdań i akapitów
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", content)

        if not sentences:
            return content

        # Score każdego zdania
        scored: list[tuple[float, str]] = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_terms = self._extract_terms(sent)
            overlap = len(query_terms & sent_terms)
            score = overlap / len(query_terms)  # query_terms nie jest pusty (sprawdzamy wyżej)
            scored.append((score, sent))

        if not scored:
            return content

        # Wybierz zdania powyżej progu trafności
        selected_sents = [sent for score, sent in scored if score >= _RELEVANCE_THRESHOLD]

        if not selected_sents:
            # Fallback: weź najlepiej ocenione zdanie żeby nie zwracać pustego wyniku
            best_score, best_sent = max(scored, key=lambda x: x[0])
            selected_sents = [best_sent]

        # Rekonstruuj w oryginalnej kolejności (nie w kolejności score)
        # Zachowanie kolejności ważne dla spójności narracji w dokumentach proceduralnych
        original_order = [sent for _, sent in scored if sent in set(selected_sents)]
        result = " ".join(original_order)

        # Przytnij do przybliżonego limitu tokenów (4 znaki/token to bezpieczne przybliżenie)
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            # Przetnij na granicy słowa zamiast w środku tokenu
            result = result[:max_chars].rsplit(" ", 1)[0] + "..."

        return result
