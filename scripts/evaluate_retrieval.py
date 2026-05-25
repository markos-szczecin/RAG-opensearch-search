#!/usr/bin/env python3
"""
CLI script: ewaluacja jakości retrieval na złotym zbiorze zapytań.

Uruchomienie:
  python scripts/evaluate_retrieval.py
  python scripts/evaluate_retrieval.py --k 10 --output results.json

Wymagania:
  - OpenSearch z zaindeksowanymi dokumentami (uruchom najpierw index_documents.py)
  - Klucz OpenAI API w .env (potrzebny do VectorSearch embeddings)

Co to jest "złoty zbiór zapytań" (golden query set)?
------------------------------------------------------
Złoty zbiór to ręcznie przygotowany zestaw par (zapytanie, oczekiwane dokumenty).
Dla każdego zapytania wiemy które dokumenty POWINNY być zwrócone jako relevantne.

Przykład:
  query:              "What is the premium account transfer limit?"
  expected_doc_ids:   ["account-limits-v1"]

Uruchamiamy każde zapytanie przez każdy tryb wyszukiwania i sprawdzamy
czy faktycznie zwrócone dokumenty to te które powinny być relevantne.

Metryki ewaluacyjne
--------------------
Wszystkie metryki obliczamy "at k" (na top-k wynikach):

P@k (Precision at k):
  Jaką część zwróconych wyników faktycznie jest relevantna?
  P@5 = 3/5 = 0.6 gdy 3 z 5 zwróconych dokumentów jest na liście oczekiwanych.

R@k (Recall at k):
  Jaką część wszystkich relevantnych dokumentów znaleźliśmy?
  R@5 = 2/3 = 0.67 gdy znaleźliśmy 2 z 3 oczekiwanych dokumentów.

MRR (Mean Reciprocal Rank):
  Na której pozycji pojawia się PIERWSZY relevantny dokument?
  MRR = 1.0 gdy relevantny jest na pozycji 1.
  MRR = 0.5 gdy relevantny jest na pozycji 2.
  Ważna metryka dla systemów gdzie użytkownik klika pierwszy wynik.

Interpretacja wyników:
  Oczekiwany ranking: hybrid > keyword > vector (dla dokumentów fintech)
  Hybrid łączy zalety obu — powinien wygrywać na większości zapytań.
  Vector może wygrać na parafrazowanych zapytaniach.
  Keyword wygra na dokładnych terminach (numery polis, nazwy produktów).

Dla MVP system jest "dobry" przy:
  P@5 > 0.5, R@5 > 0.6, MRR > 0.7

Jak interpretować latency?
  keyword:  <20ms  (tylko BM25, brak embeddings)
  vector:   <300ms (80-150ms embedding + 5-20ms kNN search)
  hybrid:   <320ms (parallel keyword + vector, potem merge)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.metrics import mrr, precision_at_k, recall_at_k

# Złoty zbiór zapytań dla dokumentów w docs_sample/
# Format: query → expected_doc_ids (podzbiór zaindeksowanych dokumentów)
# Każde zapytanie testuje inny aspekt systemu wyszukiwania.
GOLDEN_QUERIES = [
    {
        "query": "What should a user do after losing their phone?",
        "expected_doc_ids": ["mobile-auth-policy-v3"],
        "notes": "Testuje wyszukiwanie procedur bezpieczeństwa",
    },
    {
        "query": "What is the premium account transfer limit?",
        "expected_doc_ids": ["account-limits-v1"],
        "notes": "Testuje wyszukiwanie tabelaryczne (CSV → NL sentences)",
    },
    {
        "query": "How do I open a business account?",
        "expected_doc_ids": ["product-faq-v2"],
        "notes": "Testuje FAQ retrieval",
    },
    {
        "query": "What AML rules apply to transfers above 10000 EUR?",
        "expected_doc_ids": ["transfer-procedures-v2", "compliance-notes-v4"],
        "notes": "Testuje multi-document retrieval (dwa relevantne dokumenty)",
    },
    {
        "query": "How many consecutive failed PIN attempts lock the device?",
        "expected_doc_ids": ["mobile-auth-policy-v3"],
        "notes": "Testuje dokładność liczbową (konkretna wartość w dokumencie)",
    },
    {
        "query": "Can a student account make foreign transfers?",
        "expected_doc_ids": ["account-limits-v1", "transfer-procedures-v2"],
        "notes": "Testuje cross-document retrieval dla złożonych pytań",
    },
    {
        "query": "What GDPR retention period applies to transaction records?",
        "expected_doc_ids": ["compliance-notes-v4"],
        "notes": "Testuje terminologię prawną i regulatory retrieval",
    },
    {
        "query": "What fees apply to SEPA transfers?",
        "expected_doc_ids": ["product-faq-v2"],
        "notes": "Testuje wyszukiwanie cennika i produktów",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality across keyword, vector, and hybrid modes",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff rank dla P@k i R@k (domyślnie: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Zapisz wyniki JSON do tego pliku",
    )
    parser.add_argument(
        "--role",
        default="admin",
        help="Rola użytkownika dla ewaluacji (domyślnie: admin — widzi wszystkie dokumenty)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from app.dependencies import get_hybrid_search, get_keyword_search, get_vector_search
    from app.models.search import SearchFilters, SearchRequest

    # Inicjalizacja serwisów wyszukiwania z dependency injection
    keyword_svc = get_keyword_search()
    vector_svc = get_vector_search()
    hybrid_svc = get_hybrid_search()

    modes = {
        "keyword": keyword_svc,
        "vector": vector_svc,
        "hybrid": hybrid_svc,
    }

    print(f"Ewaluacja {len(GOLDEN_QUERIES)} golden queries, k={args.k}, rola={args.role}")
    print("=" * 70)

    # Wyniki per tryb: lista dict z metrykami per zapytanie
    all_results: dict[str, list[dict]] = {mode: [] for mode in modes}
    all_latencies: dict[str, list[float]] = {mode: [] for mode in modes}

    for golden in GOLDEN_QUERIES:
        query = golden["query"]
        expected = golden["expected_doc_ids"]

        for mode_name, svc in modes.items():
            request = SearchRequest(
                query=query,
                top_k=args.k,
                # admin widzi wszystkie poziomy dostępu — potrzebne by evaluacja
                # była "fair" i nie pomijała dokumentów confidential
                user_role=args.role,
                # Brak filtrów dodatkowych — testujemy domyślne zachowanie
                filters=SearchFilters(),
            )

            query_start = time.monotonic()
            response = await svc.search(request)
            query_latency = (time.monotonic() - query_start) * 1000

            # Wyciągnij doc_ids z wyników (deduplikacja — chunki z tego samego dokumentu)
            retrieved_doc_ids = list(dict.fromkeys(r.doc_id for r in response.results))

            # Oblicz metryki
            p = precision_at_k(retrieved_doc_ids, expected, args.k)
            r = recall_at_k(retrieved_doc_ids, expected, args.k)
            m = mrr(retrieved_doc_ids, expected)

            all_results[mode_name].append({
                "query": query,
                "expected": expected,
                "retrieved": retrieved_doc_ids,
                "precision_at_k": p,
                "recall_at_k": r,
                "mrr": m,
                "latency_ms": query_latency,
            })
            all_latencies[mode_name].append(query_latency)

    # Wydrukuj tabelę wyników
    print(f"\n{'Tryb':<10} {'P@k':>8} {'R@k':>8} {'MRR':>8} {'Lat p50':>10} {'Lat p95':>10}")
    print("-" * 60)

    for mode_name, rows in all_results.items():
        avg_p = sum(r["precision_at_k"] for r in rows) / len(rows)
        avg_r = sum(r["recall_at_k"] for r in rows) / len(rows)
        avg_mrr = sum(r["mrr"] for r in rows) / len(rows)

        lats = sorted(all_latencies[mode_name])
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)]

        print(
            f"{mode_name:<10} {avg_p:>8.3f} {avg_r:>8.3f} {avg_mrr:>8.3f} "
            f"{p50:>10.1f} {p95:>10.1f}"
        )

    print()

    # Pokaż szczegółowe wyniki dla każdego zapytania (tryb hybrid)
    print("Szczegółowe wyniki (hybrid):")
    print("-" * 70)
    for row in all_results["hybrid"]:
        hit = "✓" if row["mrr"] > 0 else "✗"
        print(
            f"  {hit} P@k={row['precision_at_k']:.2f} R@k={row['recall_at_k']:.2f} "
            f"MRR={row['mrr']:.2f} | {row['query'][:60]}"
        )

    # Zapisz wyniki JSON jeśli podano --output
    if args.output:
        output_data = {
            "k": args.k,
            "role": args.role,
            "n_queries": len(GOLDEN_QUERIES),
            "results": all_results,
            "summary": {
                mode: {
                    "mean_precision": sum(r["precision_at_k"] for r in rows) / len(rows),
                    "mean_recall": sum(r["recall_at_k"] for r in rows) / len(rows),
                    "mean_mrr": sum(r["mrr"] for r in rows) / len(rows),
                    "p50_latency_ms": sorted(all_latencies[mode])[len(all_latencies[mode]) // 2],
                }
                for mode, rows in all_results.items()
            },
        }
        args.output.write_text(json.dumps(output_data, indent=2, default=str))
        print(f"\nWyniki zapisane do: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
