"""
RecursiveTextChunker — dzieli dokumenty na fragmenty (chunks) ograniczone liczbą tokenów.

Dlaczego chunking jest kluczowy w systemach RAG?
-------------------------------------------------
Modele językowe mają ograniczone okno kontekstu (np. Claude Sonnet: ~200K tokenów,
ale kosztowne — dlatego budżetujemy do ~2500 tokenów kontekstu).
Wysyłanie całych dokumentów byłoby:
  1. Za drogie (tysiące tokenów na każde zapytanie)
  2. Mało precyzyjne (model "topi się" w za dużej ilości tekstu)

Chunking dzieli dokumenty na mniejsze fragmenty:
  - Każdy fragment jest semantycznie spójny (nie przerywa zdań/akapitów)
  - Możemy wybrać tylko RELEVANTNE fragmenty przez wyszukiwanie
  - Oszczędzamy tokeny i poprawiamy precyzję odpowiedzi

Dlaczego rozmiar w TOKENACH, nie znakach?
------------------------------------------
Modele językowe "myślą" tokenami, nie znakami. Jeden token ≈ 4 znaki po angielsku,
ale może to być cały wyraz lub fragment. Pomiar w znakach dałby nieprzewidywalne
rozmiary po stronie modelu. Tiktoken (ta sama biblioteka co OpenAI) daje dokładną
odpowiedź na "ile tokenów zajmie ten tekst?".

Strategia rekursywna
---------------------
Próbujemy podzielić na poziomie akapitów (\\n\\n). Jeśli akapit jest za duży,
schodzimy na poziom linii (\\n), potem słów ( ), potem znaków ("").
Najlepszy wynik: chunki z naturalnymi granicami, bez rozrywania zdań w połowie.
"""

import warnings

import tiktoken

from app.models.document import Chunk, DocumentMetadata
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimalna liczba tokenów w chunku — chunki poniżej tego progu są odrzucane.
# Dlaczego 20? Mniej niż ~20 tokenów to zwykle urwany fragment zdania bez
# samodzielnego sensu informacyjnego. Lepiej go pominąć niż indeksować jako
# bezużyteczny szum w bazie wektorowej.
MIN_CHUNK_TOKENS = 20


class RecursiveTextChunker:
    """
    Dzieli tekst na fragmenty ograniczone liczbą tokenów, z konfigurowalnymi zakładkami.

    Strategia podziału (w kolejności priorytetu):
      1. Podwójny znak nowej linii \\n\\n (granica akapitu) — preferowane
      2. Pojedynczy znak nowej linii \\n (granica linii)
      3. Spacja (granica słowa)
      4. Znak "" (ostatnia deska ratunku — unika przekroczenia budżetu tokenów)

    Implementacja odzwierciedla LangChain's RecursiveCharacterTextSplitter,
    ale jest świadoma tokenów (tiktoken), a nie znaków.

    Dlaczego zakładka (overlap)?
    ------------------------------
    Gdy tekst jest dzielony w środku zdania, zdanie zostaje podzielone między
    dwa chunki. Bez zakładki model tracałby kontekst na granicy chunków.
    Przykład: zdanie "Maksymalny limit dzienny wynosi 10 000 EUR i dotyczy
    wszystkich rachunków" może być podzielone — zakładka kopiuje końcowe 75
    tokenów chunka N na początek chunka N+1, więc żaden chunk nie zaczyna się
    od urwanego zdania.

    Kompromis: zakładka zwiększa liczbę chunków i koszt embeddingów. 75 tokenów
    (~300 znaków) to dobre balansowanie między spójnością a kosztem.
    """

    # Separators tried in order; first match that keeps chunks within budget wins.
    _SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            # Last resort: split character by character into token-sized pieces
            tokens = self._enc.encode(text)
            segments = []
            for i in range(0, len(tokens), self.chunk_size):
                segments.append(self._enc.decode(tokens[i : i + self.chunk_size]))
            return segments

        sep, *rest = separators
        parts = text.split(sep) if sep else list(text)

        results: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = self.count_tokens(part)

            if part_tokens > self.chunk_size:
                # Flush what we have so far
                if current_parts:
                    results.append(sep.join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # Recursively split the oversized segment
                if sep == "":
                    # Already at character level — force-split by tokens
                    warnings.warn(
                        f"Segment of {part_tokens} tokens exceeds chunk_size={self.chunk_size}; "
                        "force-splitting at character level.",
                        stacklevel=4,
                    )
                    results.extend(self._split_recursive(part, []))
                else:
                    results.extend(self._split_recursive(part, rest))
                continue

            # +1 for the separator that will join this part back
            join_cost = self.count_tokens(sep) if current_parts else 0
            if current_tokens + join_cost + part_tokens > self.chunk_size and current_parts:
                results.append(sep.join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(part)
            current_tokens += self.count_tokens(sep) if len(current_parts) > 1 else 0
            current_tokens += part_tokens

        if current_parts:
            results.append(sep.join(current_parts))

        return results


    def chunk(self, text: str, metadata: DocumentMetadata) -> list[Chunk]:
        """
        Split `text` into Chunk objects.

        chunk_id format: "{doc_id}::chunk-{n:03d}"
        """
        segments = self._split_recursive(text, list(self._SEPARATORS))

        chunks: list[Chunk] = []
        prev_overlap = ""

        for i, segment in enumerate(segments):
            content = (prev_overlap + " " + segment).strip() if prev_overlap else segment

            # Only discard tiny fragments when we already have at least one chunk —
            # this avoids discarding the sole chunk of a legitimately short document.
            # MIN_CHUNK_TOKENS is meant for trailing artifacts, not full documents.
            if chunks and self.count_tokens(content) < MIN_CHUNK_TOKENS:
                logger.debug("Discarding tiny chunk at index %d (%d tokens)", i, self.count_tokens(content))
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.doc_id}::chunk-{len(chunks):03d}",
                    doc_id=metadata.doc_id,
                    content=content,
                    metadata=metadata,
                )
            )

            # Compute overlap: take trailing tokens of the raw segment (not inflated content)
            overlap_tokens = self._enc.encode(segment)[-self.overlap :]
            prev_overlap = self._enc.decode(overlap_tokens)

        return chunks
