"""
Retrieval-layer guardrail.

Runs after search, before the LLM sees any document content.
Ensures only authorised, current, and appropriate chunks reach the prompt.

TODO (Phase 3):
  - Implement _check_access_level(): compare chunk.access_level against
    the caller's allowed levels (from Settings.role_access_levels[user_role]).
  - Implement _check_freshness(): reject chunks where valid_to < today.
  - Implement _check_status(): reject draft chunks unless explicitly requested.
  - Return discarded chunk IDs + reasons for the /debug/search endpoint.
"""

from datetime import date
from typing import Any

from app.config import get_settings
from app.guardrails.base import Guardrail, GuardrailResult
from app.models.search import SearchResult

_ACCESS_HIERARCHY = {"public": 0, "internal": 1, "confidential": 2}


class RetrievalGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "retrieval_guardrail"

    def check(self, payload: Any) -> GuardrailResult:
        """
        payload: tuple[list[SearchResult], str]  →  (chunks, user_role)

        Returns GuardrailResult; caller iterates chunks and calls check() per chunk.
        Use check_chunk() directly for per-chunk filtering.
        """
        chunks, user_role = payload
        settings = get_settings()
        allowed_levels = settings.role_access_levels.get(user_role, ["public"])

        rejected = [c for c in chunks if not self._is_allowed(c, allowed_levels)]
        if rejected:
            return GuardrailResult(
                passed=False,
                action="warn",
                reason=f"{len(rejected)} chunk(s) removed by access/freshness rules",
            )
        return GuardrailResult(passed=True, action="allow")

    def filter_chunks(
        self, chunks: list[SearchResult], user_role: str
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """
        Returns (allowed_chunks, rejected_chunks).

        TODO: implement _is_allowed() per chunk.
        """
        settings = get_settings()
        allowed_levels = settings.role_access_levels.get(user_role, ["public"])
        allowed, rejected = [], []
        for chunk in chunks:
            (allowed if self._is_allowed(chunk, allowed_levels) else rejected).append(chunk)
        return allowed, rejected

    def _is_allowed(self, chunk: SearchResult, allowed_levels: list[str]) -> bool:
        """
        TODO: check access_level and freshness.
        Stub returns True (allow all) — implement before production use.
        """
        # TODO: chunk.access_level in allowed_levels
        # TODO: chunk valid_to is None OR chunk valid_to >= date.today()
        return True
