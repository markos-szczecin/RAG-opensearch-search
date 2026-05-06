from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from app.models.document import Chunk, IngestionResult

# ---------------------------------------------------------------------------
# OpenSearch index mapping
# ---------------------------------------------------------------------------
# Versioned as a constant so schema decisions are explicit and reviewable.
# When the mapping changes, bump the index version in IndexManager.
# ---------------------------------------------------------------------------
INDEX_MAPPING: dict = {
    "settings": {
        "index": {
            "knn": True,            # enables HNSW approximate nearest-neighbour
            "knn.algo_param.ef_search": 512,
        },
        "analysis": {
            "analyzer": {
                "fintech_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop",
                        "english_stemmer",
                        "fintech_synonyms",
                    ],
                }
            },
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "fintech_synonyms": {
                    "type": "synonym",
                    # TODO: expand with domain synonyms, e.g. "2FA, MFA, two-factor"
                    "synonyms": [
                        "mfa, 2fa, two-factor authentication",
                        "iban, bank account number",
                    ],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            # ---- Searchable text ----
            "content": {
                "type": "text",
                "analyzer": "fintech_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"},   # exact match / aggregations
                },
            },
            "title": {
                "type": "text",
                "analyzer": "fintech_analyzer",
                "boost": 2,   # title matches count double in BM25
                "fields": {"keyword": {"type": "keyword"}},
            },
            # ---- Vector ----
            "content_vector": {
                "type": "knn_vector",
                "dimension": 1536,      # text-embedding-3-small default
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
                # TODO: set dimension from Settings.embedding_dimensions at runtime
            },
            # ---- Keyword / filter fields ----
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "department": {"type": "keyword"},
            "language": {"type": "keyword"},
            "access_level": {"type": "keyword"},
            "status": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "version": {"type": "integer"},
            # ---- Date fields (freshness filtering) ----
            "valid_from": {"type": "date"},
            "valid_to": {"type": "date"},
        }
    },
}


@dataclass
class BulkResult:
    indexed: int
    failed: int
    errors: list[str]


class OpenSearchIndexer:
    """
    Handles low-level OpenSearch write operations: index creation and bulk upsert.

    Idempotent: uses doc_as_upsert so re-running the pipeline on unchanged
    documents is safe (no duplicates, no data loss).

    TODO (Phase 1):
      - Wire embedding_dimensions from Settings into INDEX_MAPPING at __init__ time
        instead of hardcoding 1536.
      - Add delete_by_doc_id() using a delete-by-query on the doc_id field.
      - Add a refresh() call after bulk indexing if you need immediate searchability
        (at the cost of throughput during load tests).
    """

    def __init__(self, client: AsyncOpenSearch, index_name: str) -> None:
        self._client = client
        self._index = index_name

    async def bulk_upsert(self, chunks: list[Chunk]) -> BulkResult:
        """
        Upsert a batch of chunks.  Uses chunk_id as the document _id so
        re-indexing the same chunk is idempotent.

        TODO: call opensearchpy.helpers.async_bulk with the action list below.
        """
        actions = [
            {
                "_op_type": "update",
                "_index": self._index,
                "_id": chunk.chunk_id,
                "doc": chunk.to_opensearch_doc,
                "doc_as_upsert": True,
            }
            for chunk in chunks
        ]

        # TODO: uncomment when implementing Phase 1
        # success, errors = await async_bulk(self._client, actions, raise_on_error=False)
        # return BulkResult(indexed=success, failed=len(errors), errors=[str(e) for e in errors])

        raise NotImplementedError("OpenSearchIndexer.bulk_upsert() — implement in Phase 1")

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document.
        Returns the number of deleted documents.

        TODO: use delete_by_query with a term filter on doc_id.
        """
        raise NotImplementedError
