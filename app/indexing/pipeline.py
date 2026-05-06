import time
from pathlib import Path

from app.config import Settings
from app.indexing.chunker import RecursiveTextChunker
from app.indexing.embedder.base import Embedder
from app.indexing.loaders import CSVLoader, DocumentLoader, MarkdownLoader, PDFLoader
from app.indexing.opensearch_indexer import OpenSearchIndexer
from app.indexing.pii_detector import PIIDetector
from app.models.document import DocumentMetadata, IngestionResult

_LOADER_REGISTRY: list[type[DocumentLoader]] = [MarkdownLoader, PDFLoader, CSVLoader]


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow:
      load → PII redact → chunk → embed → bulk index

    Design notes:
      - Each stage is a separate concern; swap any stage without touching others.
      - Embedding is async; chunking and PII detection are sync (CPU-bound,
        fast enough without async for typical document sizes).
      - TODO (Phase 1): add progress callbacks / logging at each stage.
      - TODO (Phase 3): add a dry_run=True mode that runs everything except
        the final index write (useful for testing mapping changes).
    """

    def __init__(
        self,
        embedder: Embedder,
        indexer: OpenSearchIndexer,
        settings: Settings,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ) -> None:
        self._embedder = embedder
        self._indexer = indexer
        self._settings = settings
        self._chunker = RecursiveTextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self._pii = PIIDetector()
        self._loaders: list[DocumentLoader] = [cls() for cls in _LOADER_REGISTRY]

    def _select_loader(self, path: Path) -> DocumentLoader:
        """Return the first loader that supports this file extension."""
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise ValueError(f"No loader registered for extension '{path.suffix}'")

    async def ingest_file(self, path: Path, metadata: DocumentMetadata) -> IngestionResult:
        """
        Full pipeline for a single file.

        Step 1 — Load: parse the file into (text, metadata) sections.
        Step 2 — PII: redact detected PII before anything is stored.
        Step 3 — Chunk: split each section into token-bounded Chunk objects.
        Step 4 — Embed: call the embedder in batches; attach vectors to chunks.
        Step 5 — Index: bulk upsert all chunks to OpenSearch.
        Step 6 — Return: IngestionResult with stats.

        TODO (Phase 1): implement each step.
        TODO (Phase 3): emit structured log lines per step for observability.
        """
        start = time.monotonic()
        loader = self._select_loader(path)

        # Step 1
        sections = loader.load(path, metadata)

        # Step 2 + 3
        all_chunks = []
        for text, sec_metadata in sections:
            clean_text = self._pii.redact(text)
            chunks = self._chunker.chunk(clean_text, sec_metadata)
            all_chunks.extend(chunks)

        # Step 4 — embed in batches
        texts = [c.content for c in all_chunks]
        vectors = await self._embedder.embed_texts(texts)
        for chunk, vector in zip(all_chunks, vectors):
            chunk.content_vector = vector

        # Step 5
        result = await self._indexer.bulk_upsert(all_chunks)

        duration = time.monotonic() - start
        return IngestionResult(
            doc_id=metadata.doc_id,
            n_chunks=len(all_chunks),
            n_indexed=result.indexed,
            n_failed=result.failed,
            duration_seconds=round(duration, 2),
        )

    async def ingest_directory(
        self, directory: Path, metadata_overrides: dict[str, DocumentMetadata] | None = None
    ) -> list[IngestionResult]:
        """
        Ingest all supported files in a directory.

        metadata_overrides: maps filename → DocumentMetadata for files that
        need custom metadata. Files not in the map get minimal auto-generated
        metadata (title from filename, sensible defaults).

        TODO (Phase 1): implement auto-metadata generation.
        TODO: run files concurrently with asyncio.gather (watch rate limits).
        """
        results = []
        overrides = metadata_overrides or {}
        for path in sorted(directory.iterdir()):
            try:
                loader = self._select_loader(path)
            except ValueError:
                continue   # unsupported extension — skip silently

            meta = overrides.get(path.name)
            if meta is None:
                # TODO: generate sensible defaults from filename
                raise ValueError(f"No metadata provided for {path.name}")

            results.append(await self.ingest_file(path, meta))
        return results
