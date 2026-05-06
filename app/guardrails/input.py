"""
Input-layer guardrail.

Runs before any retrieval or LLM call.  Catches prompt injection and
clearly out-of-scope queries without spending tokens.

TODO (Phase 3 — implement each check):
  1. INJECTION_PATTERNS: block if query matches any pattern.
  2. PII_IN_QUERY: warn (don't log raw query); strip PII before logging.
  3. TOPIC_BLOCKLIST: block financial advice, medical, legal advice.
  4. MAX_LENGTH: block absurdly long queries (potential token stuffing).
"""

import re
from typing import Any

from app.guardrails.base import Guardrail, GuardrailResult
from app.indexing.pii_detector import PIIDetector

# Known prompt injection patterns.
# Expand this list as new jailbreak techniques are discovered.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|prior|all)\s+instructions", re.IGNORECASE),
    re.compile(r"(show|reveal|print|dump)\s+(all|every)\s+(internal|system)\s+(doc|document)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+(access|security|filter)", re.IGNORECASE),
]

_MAX_QUERY_LENGTH = 1000


class InputGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "input_guardrail"

    def __init__(self) -> None:
        self._pii = PIIDetector()

    def check(self, payload: Any) -> GuardrailResult:
        query: str = payload

        # --- Length check ---
        if len(query) > _MAX_QUERY_LENGTH:
            return GuardrailResult(
                passed=False,
                action="block",
                reason=f"Query exceeds maximum length of {_MAX_QUERY_LENGTH} characters",
            )

        # --- Injection detection ---
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    passed=False,
                    action="block",
                    reason="Potential prompt injection detected",
                )

        # --- PII detection ---
        if self._pii.has_pii(query):
            # Warn but don't block — caller should avoid logging raw query
            return GuardrailResult(
                passed=True,
                action="warn",
                reason="PII detected in query",
                pii_detected=True,
                safe_query=self._pii.redact(query),
            )

        return GuardrailResult(passed=True, action="allow")
