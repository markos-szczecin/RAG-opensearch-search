"""
Integration tests for the RAG graph workflow.

Requires running OpenSearch + ANTHROPIC_API_KEY environment variable.
Run with: pytest -m integration

TODO (Phase 2):
  - Test full /ask flow end-to-end with a real indexed document.
  - Test that injection attempts are blocked before retrieval.
  - Test that confidential chunks don't appear in customer answers.
  - Test multi-turn chat_history is honoured by the answer generator.
  - Test that citations in the response match retrieved chunk IDs.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_ask_returns_grounded_answer(test_client) -> None:
    """
    TODO: ask "What happens if I lose my phone?" and verify:
      - HTTP 200
      - answer is non-empty
      - citations reference mobile-auth-policy-v3
      - confidence == "grounded"
    """
    pytest.skip("TODO: implement after Phase 2 answer_generator is done")


@pytest.mark.asyncio
async def test_injection_is_blocked(test_client) -> None:
    """
    TODO: send an injection query and verify HTTP 400 with a safe error message.
    """
    pytest.skip("TODO: implement — InputGuardrail is already wired in /ask")


@pytest.mark.asyncio
async def test_customer_cannot_access_confidential_answer(test_client) -> None:
    """
    TODO: index a confidential chunk, ask as user_role=customer, and verify
    the answer does not reference that chunk.
    """
    pytest.skip("TODO: implement after permission_filter node is done")
