"""
Output-layer guardrail.

Runs after answer generation, before returning the response to the caller.
Catches hallucinations, missing citations, and metadata leakage.

TODO (Phase 3):
  - citation_present_check: fail if len(citations) == 0.
  - metadata_leak_check: scan answer for chunk_id format, valid_from dates,
    access_level strings — these should never appear in user-facing answers.
  - confidence_check: if confidence == "unsupported", force route to refusal.
  - For high-stakes queries (compliance, legal), add an optional LLM-judge
    groundedness check here (expensive — gate behind a flag).
"""

import re
from typing import Any

from app.guardrails.base import Guardrail, GuardrailResult
from app.models.rag import Citation

# Patterns that indicate internal metadata leakage
_METADATA_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"::\s*chunk-\d{3}", re.IGNORECASE),     # chunk_id format
    re.compile(r"access_level\s*[:=]", re.IGNORECASE),
    re.compile(r"valid_to\s*[:=]", re.IGNORECASE),
    re.compile(r"source_path\s*[:=]", re.IGNORECASE),
]


class OutputGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "output_guardrail"

    def check(self, payload: Any) -> GuardrailResult:
        """
        payload: tuple[str, list[Citation]]  →  (answer_text, citations)
        """
        answer: str
        citations: list[Citation]
        answer, citations = payload

        # --- Citation presence ---
        if not citations:
            return GuardrailResult(
                passed=False,
                action="warn",
                reason="Answer contains no citations; confidence downgraded",
            )

        # --- Metadata leakage ---
        for pattern in _METADATA_LEAK_PATTERNS:
            if pattern.search(answer):
                return GuardrailResult(
                    passed=False,
                    action="block",
                    reason="Answer contains internal metadata — blocked before delivery",
                )

        return GuardrailResult(passed=True, action="allow")
