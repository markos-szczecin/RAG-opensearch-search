from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GuardrailResult:
    passed: bool
    action: Literal["allow", "block", "warn"]
    reason: str | None = None
    pii_detected: bool = False
    safe_query: str | None = None   # sanitised version of the input, if modified


class Guardrail(ABC):
    """
    Abstract guardrail interface.

    Each guardrail owns one layer of safety checking (input, retrieval, or
    output).  They are deliberately sync — guardrails must be fast (< 5 ms)
    so they don't add meaningful latency to the request path.

    For expensive checks (LLM-based classification), the query_classifier node
    handles them; keep guardrails to regex + rule-based logic.
    """

    @abstractmethod
    def check(self, payload: Any) -> GuardrailResult:
        """
        Evaluate the payload against this guardrail's rules.

        Args:
            payload: The object to check.  Type varies by layer:
                     - InputGuardrail: str (raw query)
                     - RetrievalGuardrail: list[SearchResult]
                     - OutputGuardrail: tuple[str, list[Citation]]  (answer, citations)

        Returns:
            GuardrailResult with action = allow | block | warn.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...
