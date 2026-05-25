"""
Health check endpoint — weryfikuje dostępność wszystkich zależności zewnętrznych.

Dlaczego health check jest ważny w systemach RAG?
---------------------------------------------------
System RAG zależy od wielu zewnętrznych serwisów:
  - OpenSearch: bez niego wyszukiwanie zwróci puste wyniki (cichy błąd)
  - PostgreSQL: bez niego feedback nie jest zapisywany (cichy błąd)
  - OpenAI API: bez niego embeddingi są zerowe (dysfunkcjonalne wyszukiwanie wektorowe)
  - Anthropic API: bez niego generowanie odpowiedzi zwraca błąd

Health endpoint pozwala:
  1. Load balancerowi usunąć unhealthy instancje z rotacji
  2. Orchestratorowi (Kubernetes) restartować pod przy health check failure
  3. Dashboardom monitorującym wyświetlać status systemu
  4. Inżynierom szybko diagnozować który serwis zawiódł

Status "degraded" vs "down":
  "degraded": co najmniej jedna zależność jest niedostępna, ale API odpowiada
  "down":     API samo w sobie nie odpowiada (health endpoint niedostępny)

Mierzymy latency każdej zależności bo:
  - OpenSearch latency > 100ms sugeruje przeciążenie indeksu lub sieć
  - PostgreSQL latency > 50ms sugeruje problem z connection pool
  - Te wartości bazowe pomagają wykryć degradację przed właściwym błędem
"""

import time

from fastapi import APIRouter, Depends
from opensearchpy import AsyncOpenSearch

from app.dependencies import get_eval_repository, get_opensearch_client
from app.evaluation.repository import EvalRepository

router = APIRouter()


@router.get("/health")
async def health_check(
    os_client: AsyncOpenSearch = Depends(get_opensearch_client),
    eval_repo: EvalRepository = Depends(get_eval_repository),
) -> dict:
    """
    Sprawdza dostępność OpenSearch i PostgreSQL, mierząc latency każdego.

    Wykonuje dwa niezależne testy:
      1. OpenSearch ping: os_client.ping() — prosty HTTP HEAD request do /_cluster/health
      2. PostgreSQL ping: eval_repo.ping() — SELECT 1

    Zwraca status "ok" tylko gdy OBA serwisy działają.
    Status "degraded" informuje monitoring że coś wymaga uwagi.

    Returns:
        {
          "status": "ok" | "degraded",
          "opensearch": {"ok": bool, "latency_ms": float},
          "postgres": {"ok": bool, "latency_ms": float}
        }
    """
    # OpenSearch ping — mierzymy czas osobno dla profilu zależności
    os_ok = False
    os_latency_ms = -1.0
    try:
        os_start = time.monotonic()
        os_ok = await os_client.ping()
        os_latency_ms = round((time.monotonic() - os_start) * 1000, 2)
    except Exception:
        # Każdy wyjątek (ConnectionError, Timeout) → os_ok = False
        # Nie propagujemy wyjątku — health endpoint ZAWSZE musi odpowiedzieć
        pass

    # PostgreSQL ping
    pg_ok = False
    pg_latency_ms = -1.0
    try:
        pg_latency_ms = round(await eval_repo.ping(), 2)
        pg_ok = True
    except Exception:
        pass

    # "ok" tylko gdy wszystkie krytyczne zależności działają
    overall = "ok" if (os_ok and pg_ok) else "degraded"

    return {
        "status": overall,
        "opensearch": {"ok": os_ok, "latency_ms": os_latency_ms},
        "postgres": {"ok": pg_ok, "latency_ms": pg_latency_ms},
    }
