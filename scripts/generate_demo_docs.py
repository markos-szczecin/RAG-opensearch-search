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

from anthropic import AsyncAnthropic

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


_DEPT_BY_TYPE = {
    "faq": "product",
    "developer": "engineering",
    "compliance": "compliance",
    "procedure": "operations",
    "policy": "compliance",
}

_SYSTEM_PROMPT = """\
You generate fictional NovaBanque fintech policy documents in Markdown format.
NovaBanque is a fictional digital bank used strictly for demonstration purposes.

Every document MUST begin with YAML front matter using exactly this structure:
---
doc_id: <provided-slug>
title: <provided title>
doc_type: <provided doc_type>
department: <provided department>
language: en
access_level: <provided access_level>
status: approved
valid_from: 2025-01-01
version: 1
---

After the front matter, write 400–600 words of realistic but clearly fictional
content for NovaBanque. Use markdown headings, bullet points, and bold text
appropriate for the document type. Never use real company names, real people,
or actual regulatory text verbatim.\
"""


async def generate_document(topic: str, doc_type: str, access_level: str, model: str, client: AsyncAnthropic) -> str:
    slug = topic.lower().replace(" ", "-")
    department = _DEPT_BY_TYPE.get(doc_type, "operations")

    response = await client.messages.create(
        model=model,
        max_tokens=1200,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Cache the system prompt across all documents in this session.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a {doc_type} document with the following front matter values:\n"
                    f"  doc_id: {slug}-v1\n"
                    f"  title: {topic}\n"
                    f"  doc_type: {doc_type}\n"
                    f"  department: {department}\n"
                    f"  access_level: {access_level}\n\n"
                    "Then write 400–600 words of body content."
                ),
            }
        ],
    )
    return next(b.text for b in response.content if b.type == "text")


async def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topics = GENERATION_TOPICS[: args.count]
    print(f"Generating {len(topics)} documents → {args.output_dir}")

    client = AsyncAnthropic()
    for topic, doc_type, access_level in topics:
        filename = topic.lower().replace(" ", "_") + ".md"
        content = await generate_document(topic, doc_type, access_level, args.model, client)
        output_path = args.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"  [OK] {filename}")


if __name__ == "__main__":
    asyncio.run(main())
