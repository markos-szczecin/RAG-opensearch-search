"""
LangGraph node: context_budgeter

Enforces the token budget before the expensive LLM generation call.

Dlaczego budżetowanie tokenów jest kluczowe?
----------------------------------------------
Modele językowe mają ograniczone okno kontekstu i — co ważniejsze — stały koszt
za każdy token wejściowy. Wysyłanie wszystkich znalezionych chunków bez limitu:
  1. Może przekroczyć limit kontekstu modelu (błąd API)
  2. Jest droższe niż potrzeba (duże chunki = wyższy koszt)
  3. Paradoksalnie pogarsza jakość — model "rozprasza się" na za dużo tekstu
     (zjawisko "lost in the middle": środkowe fragmenty są mniej efektywnie
     przetwarzane niż początki i końce kontekstu)

Konfiguracja budżetu (z Settings):
  max_context_tokens  = 2500   łącznie na cały kontekst dokumentów
  max_chunks          = 5      twardy limit ilości fragmentów
  max_tokens_per_chunk = 400   limit per fragment po kompresji

Algorytm:
  1. Dla każdego chunku: oblicz ile tokenów zajmie wraz z wrapperem szablonu
  2. Dodawaj chunki dopóki mieścimy się w budżecie i nie przekroczyliśmy max_chunks
  3. Ostatni chunk który "nie mieści się" — przytnij do pozostałego budżetu
  4. Jeśli przycięty chunk jest za krótki (< 50 tokenów) — pomiń go

Dlaczego cl100k_base (tokenizator GPT-4) a nie tokenizator Anthropic?
------------------------------------------------------------------------
Anthropic nie udostępnia publicznego tokenizatora Python. cl100k_base (tiktoken)
zawyża liczbę tokenów Anthropic o ~5-10% — to BEZPIECZNE zawyżenie:
  - Wolelibyśmy wysłać za mały kontekst niż przekroczyć limit i dostać błąd
  - 5-10% nadmiar to tylko 125-250 tokenów przy budżecie 2500 — mały koszt
  - To samo dotyczy Haiku vs Sonnet — cl100k_base jest rozsądnym przybliżeniem

TEMPLATE_OVERHEAD_TOKENS = 30:
  CONTEXT_BLOCK_TEMPLATE zawiera stały nagłówek per chunk:
  "--- Document Excerpt [1] ---\nSource: Tytuł (doc_id)\n"
  To ~25-35 tokenów per chunk niezależnie od zawartości.
"""

import tiktoken

from app.config import get_settings
from app.models.rag import TokenUsage
from app.rag.prompts import CONTEXT_BLOCK_TEMPLATE
from app.rag.state import RAGState

# Tokenizator cl100k_base (ten sam co GPT-4 / text-embedding-3-*)
# Jest globalny bo tworzenie instancji Encoding zajmuje ~10ms (ładowanie słownika BPE).
# Reużywamy tę samą instancję dla wszystkich wywołań.
_enc = tiktoken.get_encoding("cl100k_base")

# Stały narzut tokenów za wrapper CONTEXT_BLOCK_TEMPLATE per chunk.
# Mierzony empirycznie: format "--- Document Excerpt [N] ---\nSource: X (Y)\n"
# zajmuje ~25-35 tokenów. Używamy 30 jako bezpiecznego przybliżenia.
TEMPLATE_OVERHEAD_TOKENS = 30

# Minimalna liczba tokenów contentu skróconego ostatniego chunku.
# Poniżej tego progu skrócony chunk nie wnosi wartości — lepiej go pominąć.
MIN_TRIMMED_TOKENS = 50


def _count(text: str) -> int:
    """Zwraca liczbę tokenów BPE w tekście według cl100k_base."""
    return len(_enc.encode(text))


async def context_budgeter_node(state: RAGState) -> dict:
    """
    Wybiera chunki mieszczące się w budżecie tokenów i przycina ostatni jeśli potrzeba.

    Realizuje zasadę: "Zmaksymalizuj informacje w kontekście nie przekraczając budżetu."
    Lepsza niż prosta heurystyka "weź pierwsze N chunków" bo:
      - Chunki mają różne długości (100-500 tokenów po kompresji)
      - Możemy zmieścić 6 krótkich chunków zamiast 3 długich — więcej perspektyw
      - Przycinanie ostatniego chunku zamiast odrzucania go zachowuje
        częściową informację (często najważniejsza część jest na początku akapitu)

    Args (ze state):
      compressed_chunks: Przerankowuane i skompresowane chunki z węzła reranker.

    Returns (partial state):
      budgeted_chunks: Podzbiór chunków zmieszczający się w budżecie tokenów.
      tokens:          Wstępny TokenUsage z liczbą tokenów kontekstu.
    """
    settings = get_settings()
    chunks = state.get("compressed_chunks", [])

    budget = settings.max_context_tokens
    budgeted = []
    total_tokens = 0

    for chunk in chunks:
        # Twardy limit ilości chunków — niezależnie od budżetu tokenów
        if len(budgeted) >= settings.max_chunks:
            break

        chunk_tokens = _count(chunk.content) + TEMPLATE_OVERHEAD_TOKENS

        if total_tokens + chunk_tokens <= budget:
            # Chunk mieści się w budżecie — dodaj w całości
            budgeted.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Chunk nie mieści się — spróbuj przyciąć
            remaining_budget = budget - total_tokens - TEMPLATE_OVERHEAD_TOKENS

            if remaining_budget < MIN_TRIMMED_TOKENS:
                # Za mało miejsca nawet na sensowny fragment — przerwij
                break

            # Przytnij treść chunku do pozostałego budżetu
            # _enc.encode zwraca listę int (BPE token IDs)
            # _enc.decode zamienia je z powrotem na string
            tokens = _enc.encode(chunk.content)
            trimmed_tokens = tokens[:remaining_budget]
            trimmed_content = _enc.decode(trimmed_tokens)

            # Dodaj "..." by zasygnalizować że chunk jest obcięty.
            # Model językowy powinien to respektować i nie spekulować o brakującej treści.
            # model_copy tworzy nową instancję Pydantic z zmienionym polem —
            # oryginał pozostaje niezmieniony (immutable by design w Pydantic v2).
            trimmed_chunk = chunk.model_copy(
                update={"content": trimmed_content + "..."}
            )
            budgeted.append(trimmed_chunk)
            total_tokens += remaining_budget + TEMPLATE_OVERHEAD_TOKENS
            break  # Budżet wyczerpany po przyciętym chunku

    return {
        "budgeted_chunks": budgeted,
        "tokens": TokenUsage(
            context=total_tokens,
            answer=0,
            total=total_tokens,
        ),
    }
