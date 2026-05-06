import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    pii_type: str       # e.g. "IBAN", "EMAIL", "PHONE"
    value: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Regex patterns for common PII in European fintech context
# ---------------------------------------------------------------------------
_PATTERNS: dict[str, re.Pattern] = {
    "IBAN": re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"
    ),
    "EMAIL": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"\b(\+?[\d\s\-().]{7,20})\b"
    ),
    # Polish PESEL — 11 digit national ID
    "PESEL": re.compile(r"\b\d{11}\b"),
}

_REDACTION_PLACEHOLDER = {
    "IBAN": "[IBAN REDACTED]",
    "EMAIL": "[EMAIL REDACTED]",
    "PHONE": "[PHONE REDACTED]",
    "PESEL": "[PESEL REDACTED]",
}


class PIIDetector:
    """
    Detects and redacts PII from document text before indexing.

    This is a regex-only baseline.

    TODO (Phase 3):
      - Add spaCy NER (en_core_web_sm) for person/organisation name detection.
      - Add card number patterns (Luhn-validated).
      - Log PII detections (without the actual value) for compliance audit trail.
      - Make redaction reversible for admins (store encrypted mapping separately).
      - Consider presidio (Microsoft) for a more robust PII framework.
    """

    def detect(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for pii_type, pattern in _PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(PIIMatch(pii_type, m.group(), m.start(), m.end()))
        # Sort by position so redact() can process left-to-right
        matches.sort(key=lambda x: x.start)
        return matches

    def redact(self, text: str) -> str:
        """Replace detected PII with type-labelled placeholders."""
        matches = self.detect(text)
        # Process in reverse order to keep string indices valid
        for m in reversed(matches):
            placeholder = _REDACTION_PLACEHOLDER.get(m.pii_type, "[REDACTED]")
            text = text[: m.start] + placeholder + text[m.end :]
        return text

    def has_pii(self, text: str) -> bool:
        return bool(self.detect(text))
