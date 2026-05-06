#!/usr/bin/env python3
"""
CLI script: index all sample documents into OpenSearch.

Usage:
  python scripts/index_documents.py --docs-dir /docs_sample
  python scripts/index_documents.py --docs-dir /docs_sample --chunk-size 500 --chunk-overlap 75

TODO (Phase 1):
  - Build metadata map from filenames or a sidecar metadata.json.
  - Instantiate IngestionPipeline with real embedder + indexer.
  - Call pipeline.ingest_directory() and print per-file stats.
  - Add --dry-run flag that runs chunking/PII detection but skips indexing.
  - Add --force flag to re-embed and overwrite existing chunks.
"""

import argparse
import asyncio
from datetime import date
from pathlib import Path

# Metadata for each sample document (maps filename → DocumentMetadata fields).
# In a real system, this would be driven by a config file or database.
SAMPLE_METADATA = {
    "product_faq.md": {
        "doc_id": "product-faq-v2",
        "title": "Product FAQ",
        "doc_type": "faq",
        "department": "product",
        "language": "en",
        "access_level": "public",
        "status": "approved",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "version": 2,
    },
    "mobile_auth_policy.md": {
        "doc_id": "mobile-auth-policy-v3",
        "title": "Mobile Authorization Policy",
        "doc_type": "policy",
        "department": "compliance",
        "language": "en",
        "access_level": "internal",
        "status": "approved",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "version": 3,
    },
    "account_limits.csv": {
        "doc_id": "account-limits-v1",
        "title": "Account Limits",
        "doc_type": "procedure",
        "department": "product",
        "language": "en",
        "access_level": "internal",
        "status": "approved",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "version": 1,
    },
    "transfer_procedures.md": {
        "doc_id": "transfer-procedures-v2",
        "title": "Transfer Procedures",
        "doc_type": "procedure",
        "department": "operations",
        "language": "en",
        "access_level": "internal",
        "status": "approved",
        "valid_from": date(2025, 3, 1),
        "valid_to": None,
        "version": 2,
    },
    "compliance_notes.md": {
        "doc_id": "compliance-notes-v4",
        "title": "Compliance Notes",
        "doc_type": "compliance",
        "department": "compliance",
        "language": "en",
        "access_level": "confidential",
        "status": "approved",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "version": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index sample documents into OpenSearch")
    parser.add_argument("--docs-dir", default="/docs_sample", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=75)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # TODO: import and wire real dependencies
    # from app.config import get_settings
    # from app.dependencies import get_ingestion_pipeline
    # from app.models.document import DocumentMetadata

    print(f"Indexing documents from: {args.docs_dir}")
    print(f"Chunk size: {args.chunk_size} tokens, overlap: {args.chunk_overlap} tokens")
    if args.dry_run:
        print("DRY RUN — no data will be written to OpenSearch")

    for filename, meta_dict in SAMPLE_METADATA.items():
        path = args.docs_dir / filename
        if not path.exists():
            print(f"  [SKIP] {filename} — file not found")
            continue
        print(f"  [TODO] {filename} → {meta_dict['doc_id']}")
        # TODO:
        # metadata = DocumentMetadata(source_path=str(path), **meta_dict)
        # result = await pipeline.ingest_file(path, metadata)
        # print(f"  [OK] {filename}: {result.n_chunks} chunks, {result.duration_seconds:.1f}s")

    print("\nDone. TODO: implement IngestionPipeline and uncomment real indexing calls.")


if __name__ == "__main__":
    asyncio.run(main())
