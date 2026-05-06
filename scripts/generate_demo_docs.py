#!/usr/bin/env python3
"""
CLI script: generate additional synthetic fintech documents using Anthropic Claude.

Useful for expanding the knowledge base beyond the five sample documents
without using real customer data.

Usage:
  python scripts/generate_demo_docs.py --count 5 --output-dir /docs_sample/generated

TODO (Phase 1+):
  - Call Anthropic API with a prompt instructing Claude to write realistic-but-fictional
    fintech policy documents in Markdown format.
  - Include front matter with the correct metadata fields.
  - Generate documents across all doc_types and access_levels for variety.
  - Save each document with a unique filename and doc_id.
  - Optionally use prompt caching on the system instructions to reduce cost
    when generating many documents in one session.
"""

import argparse
import asyncio
from pathlib import Path

DOC_TYPES = ["policy", "faq", "procedure", "compliance", "developer"]
ACCESS_LEVELS = ["public", "internal", "confidential"]

GENERATION_TOPICS = [
    ("Credit Card Terms and Conditions", "faq", "public"),
    ("API Authentication Guide for Developers", "developer", "internal"),
    ("Sanctions Screening Procedure", "compliance", "confidential"),
    ("Account Upgrade Procedures", "procedure", "internal"),
    ("Chargeback Policy", "policy", "public"),
    ("Two-Factor Authentication Setup Guide", "faq", "public"),
    ("Data Retention and Deletion Policy", "compliance", "confidential"),
    ("Loan Application Processing Guide", "procedure", "internal"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fintech documents")
    parser.add_argument("--count", type=int, default=3, help="Number of documents to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("docs_sample/generated"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model")
    return parser.parse_args()


async def generate_document(topic: str, doc_type: str, access_level: str, model: str) -> str:
    """
    Call Anthropic API to generate a single document.

    TODO:
      - Import AsyncAnthropic and call messages.create().
      - Use a system prompt that specifies Markdown + front matter format.
      - Ask for 400–600 words of realistic but clearly fictional content.
      - Return the raw Markdown string.
    """
    # TODO: implement real generation
    return f"""---
doc_id: generated-{topic.lower().replace(' ', '-')}-v1
title: {topic}
doc_type: {doc_type}
department: generated
language: en
access_level: {access_level}
status: draft
valid_from: 2025-01-01
version: 1
---

# {topic}

TODO: Generate content using Anthropic API.
Model: {model}
"""


async def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topics = GENERATION_TOPICS[: args.count]
    print(f"Generating {len(topics)} documents → {args.output_dir}")

    for topic, doc_type, access_level in topics:
        filename = topic.lower().replace(" ", "_") + ".md"
        content = await generate_document(topic, doc_type, access_level, args.model)
        output_path = args.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"  [OK] {filename}")

    print("\nTODO: implement real Anthropic API call in generate_document()")


if __name__ == "__main__":
    asyncio.run(main())
