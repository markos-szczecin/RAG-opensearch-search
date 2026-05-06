#!/usr/bin/env python3
"""
CLI script: run the golden query set against all three retrieval modes and print a metrics table.

Usage:
  python scripts/evaluate_retrieval.py
  python scripts/evaluate_retrieval.py --output results.json

Golden query set format (golden_queries.json):
  [
    {
      "query": "What should a user do after losing their phone?",
      "expected_doc_ids": ["mobile-auth-policy-v3"],
      "expected_chunks_contain": ["Device Lockout"]  // optional: substring check
    },
    ...
  ]

TODO (Phase 4):
  - Load golden_queries.json from disk.
  - For each query, run all three search modes.
  - Compute precision@5, recall@5, MRR, and AP per query per mode.
  - Aggregate to mean metrics (MAP, mean MRR).
  - Print a formatted table and optionally write JSON.
  - Also measure and report latency percentiles (p50, p95).
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.evaluation.metrics import mrr, precision_at_k, recall_at_k

GOLDEN_QUERIES = [
    {
        "query": "What should a user do after losing their phone?",
        "expected_doc_ids": ["mobile-auth-policy-v3"],
    },
    {
        "query": "What is the premium account transfer limit?",
        "expected_doc_ids": ["account-limits-v1"],
    },
    {
        "query": "How do I open a business account?",
        "expected_doc_ids": ["product-faq-v2"],
    },
    {
        "query": "What AML rules apply to transfers above 10000 EUR?",
        "expected_doc_ids": ["transfer-procedures-v2", "compliance-notes-v4"],
    },
    {
        "query": "How many consecutive failed PIN attempts lock the device?",
        "expected_doc_ids": ["mobile-auth-policy-v3"],
    },
    {
        "query": "Can a student account make foreign transfers?",
        "expected_doc_ids": ["account-limits-v1", "transfer-procedures-v2"],
    },
    {
        "query": "What GDPR retention period applies to transaction records?",
        "expected_doc_ids": ["compliance-notes-v4"],
    },
    {
        "query": "What fees apply to SEPA transfers?",
        "expected_doc_ids": ["product-faq-v2"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank for P@k and R@k")
    parser.add_argument("--output", type=Path, help="Write JSON results to this file")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    print(f"Evaluating {len(GOLDEN_QUERIES)} golden queries at k={args.k}")
    print("=" * 70)

    # TODO: instantiate keyword, vector, hybrid search services
    # TODO: for each query, run all three modes and compute metrics

    # Stub: demonstrate metric functions with synthetic data
    example_retrieved = ["mobile-auth-policy-v3", "product-faq-v2", "account-limits-v1"]
    example_relevant = ["mobile-auth-policy-v3"]

    p = precision_at_k(example_retrieved, example_relevant, k=args.k)
    r = recall_at_k(example_retrieved, example_relevant, k=args.k)
    m = mrr(example_retrieved, example_relevant)
    print(f"Example (stub): P@{args.k}={p:.3f}, R@{args.k}={r:.3f}, MRR={m:.3f}")
    print("\nTODO: implement full evaluation loop in Phase 4")


if __name__ == "__main__":
    asyncio.run(main())
