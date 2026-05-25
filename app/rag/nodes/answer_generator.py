"""
LangGraph node: answer_generator

Wywołuje Anthropic Claude do generowania ugruntowanej odpowiedzi na podstawie
budgeted_chunks — fragmentów dokumentów wyselekcjonowanych przez poprzednie węzły.

To jest najdroższy i najbardziej skomplikowany węzeł w całym pipeline.
Dokumentacja jest szczegółowa bo decyzje projektowe tutaj mają duży wpływ
na jakość, koszt i latency systemu.

Dlaczego Claude (Anthropic) a nie OpenAI GPT?
-----------------------------------------------
W tej implementacji używamy już OpenAI do embeddings. Używamy Anthropic do
generowania bo:
  1. Claude jest znany z "constitutional AI" — ma wbudowane mechanizmy odmowy
     udzielania szkodliwych informacji bez dodatkowego fine-tuningu
  2. Prompt caching Anthropic ma prostszy model cenowy niż OpenAI (stały koszt
     za tworzenie cache, 0.1x koszt za trafienie)
  3. Claude dobrze radzi sobie z "grounded generation" — cytowaniem źródeł
     i odmową gdy odpowiedź nie jest w dostarczonych dokumentach

Prompt Caching — jak działa i dlaczego jest tutaj kluczowy
-------------------------------------------------------------
Anthropic oferuje "prompt caching" — możliwość cachowania fragmentów promptu
po stronie serwera Anthropic z TTL 5 minut.

Bez cachowania każde zapytanie:
  system_prompt (300 tokenów) + context (2500 tokenów) + query (50 tokenów)
  = 2850 tokenów WEJŚCIOWYCH za każde zapytanie
  Koszt: 2850 × $3 / 1M = $0.00855 per zapytanie

Z cachowaniem (przy 5 zapytaniach w 5-minutowym oknie do tych samych dokumentów):
  - 1. zapytanie: 2850 tokenów INPUT (tworzenie cache) = $0.00855
  - 2-5. zapytanie: 50 tokenów INPUT + 2800 tokenów CACHED ($0.30/1M zamiast $3/1M)
  = 50 × $3/1M + 2800 × $0.30/1M = $0.00015 + $0.00084 = $0.00099 per zapytanie
  Oszczędność: ~88% przy gorącym cache

Co cachujemy vs co NIE cachujemy?
  - CACHUJEMY: system prompt (stały dla każdego zapytania) ✓
  - CACHUJEMY: context block (chunki dokumentów — zmieniają się per query, ale
    przy podobnych zapytaniach o te same dokumenty cache trafia)
  - NIE cachujemy: samo pytanie użytkownika (unikalne per zapytanie) ✗

Kluczowa technika: context block jest OSOBNYM content blokiem przed pytaniem.
Dzięki temu cache key dla context jest stabilny niezależnie od pytania.
Gdyby pytanie było sklejone z kontekstem w jednym stringu, cache trafiałby
tylko gdy pytanie jest identyczne — prawie nigdy.

Struktura promptu z cachowaniem:
  messages=[{
    "role": "user",
    "content": [
      {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},
      {"type": "text", "text": query_instruction}  ← NIE cachowany
    ]
  }]

Cytowania — jak działają?
---------------------------
System prompt instruuje model: "cytuj każde zdanie faktualne jako [doc_id]".
Po wygenerowaniu odpowiedzi, funkcja _extract_citations() skanuje tekst
regex'em \[([^\]]+)\] i mapuje znalezione doc_id'y na SearchResult obiekty.

Przykład odpowiedzi modelu:
  "Dzienny limit przelewu dla konta premium wynosi 50 000 EUR [account-limits]."

_extract_citations() zwróci Citation z doc_id="account-limits", title="Account Limits"
(sklejoną z odpowiadającym chunkiem z budgeted_chunks).
"""

import re

import anthropic

from app.config import get_settings
from app.models.rag import Citation, TokenUsage
from app.rag.prompts import (
    ANSWER_GENERATOR_PROMPT,
    CONTEXT_BLOCK_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.rag.state import RAGState

# Module-level singleton dla klienta Anthropic.
# HTTP connection pool w kliencie powinien być reużywany między zapytaniami.
# Inicjalizacja przy pierwszym zapytaniu (lazy) — nie blokuje startu aplikacji.
_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Zwraca singleton klienta Anthropic (lazy initialization)."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=get_settings().anthropic_api_key
        )
    return _client


# Instrukcje stylu odpowiedzi per answer_mode.
# Dodawane na końcu promptu użytkownika (po pytaniu).
# "brief" = domyślny tryb dla szybkich odpowiedzi (czat, FAQ)
# "detailed" = dogłębna analiza (agent wsparcia technicznego)
# "step_by_step" = numerowane kroki (procedury, onboarding)
# "agent" = łańcuch myślenia przed odpowiedzią (złożone pytania compliance)
_MODE_INSTRUCTIONS: dict[str, str] = {
    "brief": "Be concise: answer in 2-3 sentences maximum.",
    "detailed": "Be thorough: cover all relevant aspects from the provided excerpts.",
    "step_by_step": "Structure your answer as numbered steps (1., 2., 3.).",
    "agent": "Think step by step before answering. Show your reasoning briefly.",
}


def _build_context_block(state: RAGState) -> str:
    """
    Buduje blok kontekstu z budgeted_chunks dla promptu.

    Każdy chunk jest zawijany w CONTEXT_BLOCK_TEMPLATE który zawiera:
      - Numer (do referencji w tekście: "Excerpt [1]")
      - Tytuł dokumentu (czytelna nazwa dla cytowań)
      - doc_id (identyfikator do parsowania cytowań przez _extract_citations)
      - Treść chunku (właściwa informacja)

    Numery [1], [2], etc. pozwalają modelowi na łatwą referencję do źródeł
    ("Based on Excerpt [2]..."). doc_id w nawiasach umożliwia parsowanie
    cytowań po fakcie.
    """
    chunks = state.get("budgeted_chunks", [])
    parts = [
        CONTEXT_BLOCK_TEMPLATE.format(
            index=i + 1,
            title=c.title,
            doc_id=c.doc_id,
            content=c.content,
        )
        for i, c in enumerate(chunks)
    ]
    return "\n".join(parts)


def _extract_citations(answer: str, state: RAGState) -> list[Citation]:
    """
    Parsuje cytowania [doc_id] z tekstu odpowiedzi i mapuje na SearchResult.

    Wzorzec: \[([^\]]+)\] — dopasowuje wszystko w nawiasach kwadratowych.
    Przykład: "limit wynosi 50 000 EUR [account-limits]" → doc_id = "account-limits"

    Deduplikacja przez dict.fromkeys() zachowuje kolejność pierwszego wystąpienia.
    To ważne — cytowania powinny być w kolejności pojawienia się w odpowiedzi,
    nie w kolejności listy chunków.

    Dopasowanie jest DOKŁADNE — doc_id musi być identyczny z chunk.doc_id.
    Jeśli model napisze "[account_limits]" zamiast "[account-limits]", nie zostanie
    dopasowany. Rozważenie normalizacji (lowercase, _ → -) jako ulepszenie.
    """
    chunks_by_doc = {c.doc_id: c for c in state.get("budgeted_chunks", [])}
    cited_ids = re.findall(r"\[([^\]]+)\]", answer)
    citations = []
    for doc_id in dict.fromkeys(cited_ids):  # zachowaj kolejność, usuń duplikaty
        if doc_id in chunks_by_doc:
            c = chunks_by_doc[doc_id]
            citations.append(
                Citation(
                    doc_id=c.doc_id,
                    title=c.title,
                    chunk_id=c.chunk_id,
                    score=c.score,
                    source_path=c.source_path,
                )
            )
    return citations


async def answer_generator_node(state: RAGState) -> dict:
    """
    Wywołuje Anthropic Claude i generuje odpowiedź z cytowaniami.

    Prompt składa się z 3 warstw (w kolejności w request do API):
      1. system: SYSTEM_PROMPT z instrukcjami zachowania modelu
                 → opatrzony cache_control (stały dla całej sesji aplikacji)
      2. context_block: treść dokumentów (budgeted_chunks)
                        → opatrzony cache_control (stały gdy te same dokumenty)
      3. query_instruction: pytanie + instrukcja stylu
                            → BEZ cache_control (zmienia się per zapytanie)

    Obsługa błędów:
      anthropic.APIError obejmuje: AuthenticationError, RateLimitError,
      APIConnectionError, APIStatusError. Przy błędzie zwracamy graceful
      error message zamiast rzucania wyjątku — użytkownik dostaje informację
      zamiast pustej strony z błędem 500.

    Args (ze state):
      budgeted_chunks: Chunki wybrane przez context_budgeter.
      query:          Zapytanie użytkownika.
      answer_mode:    Styl odpowiedzi (brief/detailed/step_by_step/agent).
      tokens:         Dotychczasowy TokenUsage (zostanie zaktualizowany).

    Returns (partial state):
      answer:     Tekst odpowiedzi.
      citations:  Lista Citation obiektów dla cytowanych dokumentów.
      tokens:     Zaktualizowany TokenUsage z rzeczywistymi kosztami.
      error:      String z komunikatem błędu (None jeśli sukces).
    """
    settings = get_settings()
    client = _get_client()
    context_block = _build_context_block(state)
    answer_mode = state.get("answer_mode", "brief")
    mode_instruction = _MODE_INSTRUCTIONS.get(answer_mode, _MODE_INSTRUCTIONS["brief"])

    existing_tokens = state.get("tokens", TokenUsage(context=0, answer=0, total=0))

    try:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.max_answer_tokens,
            # System prompt z cache_control — cachowany po stronie Anthropic.
            # TTL cache: 5 minut. Po tym czasie cache jest unieważniany.
            # "ephemeral" = tymczasowy cache (w przeciwieństwie do przyszłych
            # trybów persistent).
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(answer_mode=answer_mode),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Context block jako OSOBNY content block z cache_control.
                        # Kluczowe: context jest PRZED pytaniem i jest cachowany osobno.
                        # Gdy to samo zapytanie (lub podobne) trafia do tych samych
                        # dokumentów, ten blok będzie trafiony z cache.
                        {
                            "type": "text",
                            "text": context_block,
                            "cache_control": {"type": "ephemeral"},
                        },
                        # Pytanie użytkownika BEZ cache_control — zmienia się per request.
                        # Łączymy pytanie z instrukcją stylu w jednym bloku.
                        {
                            "type": "text",
                            "text": (
                                ANSWER_GENERATOR_PROMPT.format(
                                    # context_block jest już osobnym blokiem powyżej,
                                    # więc tutaj przekazujemy pusty string
                                    context_block="",
                                    query=state["query"],
                                )
                                + f"\n\n{mode_instruction}"
                            ),
                        },
                    ],
                }
            ],
        )

        answer = response.content[0].text
        citations = _extract_citations(answer, state)

        # Pobierz rzeczywiste użycie tokenów z odpowiedzi API.
        # response.usage zawiera:
        #   input_tokens:               tokeny wejściowe (system + context + query)
        #   output_tokens:              tokeny wyjściowe (answer)
        #   cache_creation_input_tokens: tokeny które zostały zapisane do cache (koszt 1.25x)
        #   cache_read_input_tokens:    tokeny odczytane z cache (koszt 0.1x)
        # Ostatnie dwa pola pozwalają mierzyć efektywność cachowania.
        answer_tokens = response.usage.output_tokens
        total = existing_tokens.context + answer_tokens

        return {
            "answer": answer,
            "citations": citations,
            "tokens": TokenUsage(
                context=existing_tokens.context,
                answer=answer_tokens,
                total=total,
            ),
            "error": None,
        }

    except anthropic.APIError as exc:
        # Graceful degradation: zamiast HTTP 500 zwróć informacyjny komunikat.
        # System RAG powinien działać częściowo nawet gdy LLM jest niedostępny —
        # możemy przynajmniej pokazać user'owi wyniki wyszukiwania.
        error_msg = f"Błąd generowania odpowiedzi: {type(exc).__name__}"
        return {
            "answer": (
                "Nie udało się wygenerować odpowiedzi z powodu błędu serwisu. "
                "Proszę spróbować ponownie za chwilę."
            ),
            "citations": [],
            "tokens": existing_tokens,
            "error": error_msg,
        }
