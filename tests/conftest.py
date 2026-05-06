"""
Shared pytest fixtures.

Fixtures defined here are available to all tests without explicit import.

TODO (Phase 1):
  - mock_embedder: returns deterministic zero-vectors for any input; avoids
    OpenAI API calls in unit and most integration tests.
  - opensearch_client: points at localhost:9200; used only in integration tests
    tagged with @pytest.mark.integration.
  - test_app: TestClient wrapping create_app() with dependency overrides so
    unit tests can call API endpoints without real services.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock

from app.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Override settings with safe test values."""
    return Settings(
        opensearch_host="localhost",
        opensearch_port=9200,
        opensearch_index="test_fintech_kb",
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        postgres_dsn="postgresql+asyncpg://rag:secret@localhost:5432/rag_test",
    )


@pytest.fixture
def mock_embedder():
    """
    Async mock embedder that returns deterministic zero-vectors.
    Prevents any OpenAI API calls in unit tests.
    """
    embedder = AsyncMock()
    embedder.dimensions = 1536
    embedder.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.0] * 1536 for _ in texts]
    )
    embedder.embed_query = AsyncMock(return_value=[0.0] * 1536)
    return embedder


@pytest.fixture
async def opensearch_client():
    """
    Real AsyncOpenSearch client pointing at localhost:9200.
    Only used in integration tests.

    TODO: skip automatically if OpenSearch is not reachable.
    """
    from opensearchpy import AsyncOpenSearch
    client = AsyncOpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
    yield client
    await client.close()


@pytest.fixture
async def test_client():
    """
    Async HTTPX client wrapping the FastAPI app.
    Uses dependency_overrides to inject mocks.

    TODO: override get_opensearch_client, get_embedder, get_eval_repository.
    """
    from app.main import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
