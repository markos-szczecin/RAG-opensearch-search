"""
LangGraph node: reranker

Przerankowuje filtered_chunks i stosuje kompresję kontekstową, produkując
compressed_chunks — mniejszy, wyższej jakości zbiór gotowy do budżetowania tokenów.

Dlaczego rerankowanie jest potrzebne skoro OpenSearch już rankinguje?
----------------------------------------------------------------------
OpenSearch rankinguje na podstawie trafności tekstowej (BM25) i podobieństwa
wektorowego. Te sygnały są dobre, ale nie uwzględniają czynników biznesowych:

  1. Autorytatywność: dokument ze statusem "approved" jest ważniejszy niż "draft"
  2. Świeżość: nowsza wersja polityki jest ważniejsza niż stara
  3. Wersjonowanie: V3 polityki jest bardziej aktualna niż V1

Przykład problemu bez rerankowania:
  Zapytanie: "limit przelewu dla konta premium"
  Wyniki OpenSearch: [draft_v1 (score 0.95), approved_v3 (score 0.88)]
  Po rerankowaniu: [approved_v3 (score 0.98), draft_v1 (score 0.95)]

Reranker "poprawia" ordering by odzwierciedlał wartość biznesową, nie tylko
dopasowanie tekstowe.

Opcje implementacji (od prostej do złożonej):
  1. Heurystyczne boosty (ta implementacja): bez zależności zewnętrznych,
     deterministyczne, łatwe do debugowania. Dobry punkt wyjścia.
  2. Cross-encoder (sentence-transformers): ~90MB model lokalnie, dokładniejszy
     ale wolniejszy (10-50ms per chunk). Najlepszy stosunek jakość/koszt.
  3. LLM-based ranking: wysoka dokładność, ale dodaje ~200ms i koszt API call.
     Uzasadniony tylko dla zapytań o wysokim priorytecie (np. compliance).

Ograniczenie aktualnej implementacji:
  SearchResult nie zawiera pól 'status' i 'version'.
  FilterBuilder gwarantuje że wszystkie chunki mają status="approved",
  więc boost za status jest stały (+0.1 dla wszystkich).
  Boost za wersję jest placeholder (0.0) dopóki 'version' nie trafi do SearchResult.
  TODO: dodaj status i version do SearchResult._parse_response() w keyword.py i vector.py.
"""

import datetime

from app.config import get_settings
from app.rag.compressor import ContextualCompressor
from app.rag.state import RAGState
from app.models.search import SearchResult

# Singleton kompresora — tworzony raz, reużywany dla każdego zapytania.
# Kompresja keyword overlap nie ma stanu wewnętrznego, więc jest bezpieczna
# do współdzielenia między wywołaniami.
_compressor = ContextualCompressor(strategy="keyword")


def _compute_rerank_score(chunk: SearchResult) -> float:
    """
    Oblicza wzbogacony score chunku dla rerankowania.

    Formuła:
      score_final = score_retrieval + boost_status + boost_recency

    boost_status (+0.1):
      Wszystkie chunki docierające do rerankera przeszły FilterBuilder
      który filtruje status="approved". Dlatego wszystkie są approved
      i dostają stały bonus +0.1 za autorytatywność.

      W przyszłości gdy SearchResult będzie miał pole 'status':
        boost_status = 0.1 if chunk.status == "approved" else 0.0

    boost_version (+0.05 per version):
      Wyższa wersja dokumentu = bardziej aktualna treść.
      Nie jest aktualnie obliczana bo SearchResult nie ma pola 'version'.
      Placeholder 0.0.

      W przyszłości gdy SearchResult będzie miał pole 'version':
        boost_version = min(chunk.version or 0, 10) * 0.05

    Dlaczego używamy score_retrieval jako punkt startowy a nie resetujemy do 0?
    -----------------------------------------------------------------------------
    Zachowanie oryginalnego score jako bazy oznacza że rerankowanie jest
    MODULACJĄ wyniku wyszukiwania, a nie całkowitym zastąpieniem. Chunki
    z wysokim podobieństwem semantycznym wciąż wygrywają z dokumentami
    o niskim podobieństwie ale wysokich metadanych.
    """
    base_score = chunk.score

    # Boost za autorytatywność (wszystkie chunki są approved po FilterBuilder)
    boost_status = 0.1

    # Boost za wersję — placeholder do momentu gdy SearchResult będzie miał version
    boost_version = 0.0

    return base_score + boost_status + boost_version


async def reranker_node(state: RAGState) -> dict:
    """
    Przerankowuje chunki i stosuje kompresję kontekstową.

    Kroki:
      1. Oblicz rerank score dla każdego chunku (base + boosty)
      2. Posortuj malejąco po rerank score
      3. Zachowaj top retrieve_top_k chunków (ograniczenie ilości przed kompresją)
      4. Zastosuj ContextualCompressor (skróć każdy chunk do relevantnych zdań)

    Dlaczego ograniczamy do retrieve_top_k przed kompresją?
    ---------------------------------------------------------
    Kompresja kontekstowa jest kosztowna (tokenizacja, scoring zdań).
    Stosowanie jej do 20 chunków zamiast 5 byłoby 4x droższe bez proporcjonalnej
    poprawy jakości — chunki na pozycjach 6-20 rzadko wnoszą coś do odpowiedzi.

    Po kompresji context_budgeter decyduje które chunki zmieszczą się w budżecie.

    Args (ze state):
      filtered_chunks: Chunki po filtrze uprawnień.
      query:           Oryginalne zapytanie (potrzebne dla kompresji keyword overlap).

    Returns (partial state):
      compressed_chunks: Przerankowuane i skrócone chunki gotowe do budżetowania.
    """
    settings = get_settings()
    chunks = list(state.get("filtered_chunks", []))
    query = state["query"]

    if not chunks:
        return {"compressed_chunks": []}

    # Krok 1+2: Oblicz score i posortuj
    scored = [(chunk, _compute_rerank_score(chunk)) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Krok 3: Zachowaj top retrieve_top_k przed kompresją
    # retrieve_top_k (domyślnie 5) to ile chunków trafia do LLM kontekstu.
    # Pobieramy trochę więcej (retrieve_top_k) żeby kompresja miała z czego wybierać.
    top_chunks = [chunk for chunk, _ in scored[: settings.retrieve_top_k]]

    # Krok 4: Kompresja kontekstowa
    # Dla każdego chunku wyciągamy tylko zdania relevantne do zapytania.
    # To redukuje tokeny per chunk i pozwala zmieścić więcej chunków w budżecie.
    compressed = await _compressor.compress(
        top_chunks, query, max_tokens_per_chunk=settings.max_tokens_per_chunk
    )

    return {"compressed_chunks": compressed}
