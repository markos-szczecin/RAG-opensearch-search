#!/usr/bin/env python3
"""
CLI script: indeksuje przykładowe dokumenty do OpenSearch.

Uruchomienie:
  # Zaindeksuj wszystkie dokumenty z docs_sample/
  python scripts/index_documents.py --docs-dir docs_sample

  # Podgląd bez zapisu (dry run)
  python scripts/index_documents.py --docs-dir docs_sample --dry-run

  # Własny rozmiar chunku
  python scripts/index_documents.py --docs-dir docs_sample --chunk-size 400 --chunk-overlap 60

Wymagania:
  - OpenSearch działający na localhost:9200 (lub zgodnie z .env)
  - Klucz OpenAI API w .env (OPENAI_API_KEY)
  - Uruchom z katalogu głównego projektu: cd <project-root>

Co robi ten skrypt?
--------------------
1. Inicjalizuje wszystkie zależności (klient OS, embedder, pipeline)
2. Tworzy indeks OpenSearch jeśli nie istnieje (z mapowaniem kNN + BM25)
3. Dla każdego pliku w SAMPLE_METADATA:
   a. Wczytuje plik przez odpowiedni loader (Markdown/PDF/CSV)
   b. Redaguje PII z treści
   c. Dzieli na chunki (token-bounded, z zakładką)
   d. Generuje embeddingi przez OpenAI API (z batching i retry)
   e. Zapisuje chunki do OpenSearch (bulk upsert)
4. Drukuje statystyki: liczba chunków, czas, status

Dlaczego metadane są hardcoded tutaj a nie w plikach?
------------------------------------------------------
W produkcji metadane byłyby w osobnym rejestrze (np. tabela SQL z listą dokumentów
i ich atrybutami). Na potrzeby demo mamy je tutaj jako stały słownik.
Alternatywa: plik metadata.yaml w katalogu docs_sample/ — łatwiejszy w utrzymaniu.
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import date
from pathlib import Path

# Dodaj root projektu do PYTHONPATH żeby importy app.* działały
# gdy skrypt jest uruchamiany z dowolnego katalogu
sys.path.insert(0, str(Path(__file__).parent.parent))


# Metadane dla każdego dokumentu przykładowego.
# Format: {nazwa_pliku: dict z polami DocumentMetadata}
# doc_id: unikalny identyfikator dokumentu w systemie (użyty jako klucz wyszukiwania)
# access_level: "public" | "internal" | "confidential" | "restricted"
SAMPLE_METADATA: dict[str, dict] = {
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
    parser = argparse.ArgumentParser(
        description="Index sample fintech documents into OpenSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs-dir",
        default="docs_sample",
        type=Path,
        help="Katalog z dokumentami do zaindeksowania (domyślnie: docs_sample)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maksymalny rozmiar chunku w tokenach (domyślnie: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=75,
        help="Liczba tokenów zakładki między chunkami (domyślnie: 75)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Uruchom pipeline bez zapisu do OpenSearch (podgląd)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from app.config import get_settings
    from app.dependencies import get_index_manager, get_ingestion_pipeline
    from app.models.document import DocumentMetadata

    settings = get_settings()

    print(f"Indeksowanie dokumentów z: {args.docs_dir.resolve()}")
    print(f"OpenSearch: {settings.opensearch_host}:{settings.opensearch_port}/{settings.opensearch_index}")
    print(f"Model embeddings: {settings.openai_embedding_model} ({settings.embedding_dimensions} wymiarów)")
    print(f"Rozmiar chunku: {args.chunk_size} tokenów, zakładka: {args.chunk_overlap} tokenów")
    if args.dry_run:
        print("TRYB DRY RUN — żadne dane nie zostaną zapisane do OpenSearch")
    print()

    # Krok 1: Upewnij się że indeks istnieje (idempotentne)
    index_manager = get_index_manager()
    if not args.dry_run:
        created = await index_manager.create_index_if_not_exists()
        if created:
            print(f"[OK] Utworzono nowy indeks: {settings.opensearch_index}")
        else:
            print(f"[OK] Indeks {settings.opensearch_index} już istnieje")

    # Krok 2: Pobierz pipeline z dependency injection
    # Pipeline używa lru_cache — te same instancje embeddera i indexera co aplikacja
    pipeline = get_ingestion_pipeline()

    total_chunks = 0
    total_indexed = 0
    total_failed = 0
    start_all = time.monotonic()

    for filename, meta_dict in SAMPLE_METADATA.items():
        path = args.docs_dir / filename

        if not path.exists():
            print(f"  [SKIP] {filename} — plik nie istnieje")
            continue

        metadata = DocumentMetadata(source_path=str(path.resolve()), **meta_dict)
        doc_id = meta_dict["doc_id"]

        if args.dry_run:
            # W trybie dry run: załaduj i podziel na chunki, ale nie embedduj ani nie indeksuj
            print(f"  [DRY RUN] {filename} → {doc_id}")
            continue

        try:
            file_start = time.monotonic()
            result = await pipeline.ingest_file(path, metadata)
            elapsed = time.monotonic() - file_start

            status = "OK" if result.n_failed == 0 else "PARTIAL"
            print(
                f"  [{status}] {filename}: "
                f"{result.n_chunks} chunków, "
                f"{result.n_indexed} zaindeksowanych, "
                f"{result.n_failed} błędów, "
                f"{elapsed:.1f}s"
            )

            total_chunks += result.n_chunks
            total_indexed += result.n_indexed
            total_failed += result.n_failed

        except Exception as exc:
            print(f"  [ERROR] {filename}: {exc}")

    elapsed_all = time.monotonic() - start_all

    print()
    print("=" * 60)
    print(f"Łącznie: {total_chunks} chunków, {total_indexed} zaindeksowanych, "
          f"{total_failed} błędów, {elapsed_all:.1f}s")

    if total_failed > 0:
        print(f"UWAGA: {total_failed} chunków nie zostało zaindeksowanych.")
        print("Sprawdź logi OpenSearch dla szczegółów błędów.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
