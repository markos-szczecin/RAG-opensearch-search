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
    Liveness + dependency check endpoint.

    TODO (Phase 1):
      - Ping OpenSearch: await os_client.ping()
      - Ping Postgres: run a trivial SELECT 1 via eval_repo
      - Return latency per dependency for monitoring dashboards.
    """
    # TODO: replace stubs with real pings
    return {
        "status": "ok",
        "opensearch": True,   # TODO: await os_client.ping()
        "postgres": True,     # TODO: await eval_repo.ping()
    }
