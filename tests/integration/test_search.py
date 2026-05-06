"""
Integration tests for search services.

Requires a running OpenSearch instance at localhost:9200.
Run with: pytest -m integration

TODO (Phase 1):
  - Create a test index with a small set of known chunks before each test.
  - Verify keyword search returns results for exact term matches.
  - Verify vector search returns semantically similar results.
  - Verify hybrid search combines both and returns expected top result.
  - Clean up (delete) the test index after each test session.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_keyword_search_returns_results(opensearch_client) -> None:
    """
    TODO: index a test chunk containing "mobile authorization" and verify
    that a keyword search for "mobile authorization" returns it.
    """
    pytest.skip("TODO: implement after Phase 1 indexing is complete")


@pytest.mark.asyncio
async def test_vector_search_semantic_match(opensearch_client, mock_embedder) -> None:
    """
    TODO: index a chunk about "lost phone procedure" and verify that
    a semantic query "what to do when device is stolen" retrieves it.
    """
    pytest.skip("TODO: implement after Phase 1 vector indexing is complete")


@pytest.mark.asyncio
async def test_hybrid_outperforms_single_mode(opensearch_client, mock_embedder) -> None:
    """
    TODO: on a set of ambiguous queries, verify that hybrid search recall@5
    is >= max(keyword recall@5, vector recall@5).
    """
    pytest.skip("TODO: implement in Phase 2 evaluation work")


@pytest.mark.asyncio
async def test_access_level_filtering(opensearch_client, mock_embedder) -> None:
    """
    TODO: index chunks with access_level=confidential and verify that
    a search with user_role=customer does NOT return those chunks.
    """
    pytest.skip("TODO: implement after FilterBuilder access level logic is done")
