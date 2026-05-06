"""
Unit tests for RecursiveTextChunker.

TODO (Phase 1): implement tests once the chunker is implemented.
"""

import pytest

from app.indexing.chunker import RecursiveTextChunker
from app.models.document import DocumentMetadata
from datetime import date


@pytest.fixture
def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="test-doc-v1",
        title="Test Document",
        doc_type="faq",
        department="test",
        language="en",
        access_level="public",
        status="approved",
        valid_from=date(2025, 1, 1),
        source_path="test.md",
        version=1,
    )


def test_chunk_id_format(metadata: DocumentMetadata) -> None:
    """chunk_id must follow the '{doc_id}::chunk-{n:03d}' format."""
    chunker = RecursiveTextChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk("Hello world. " * 10, metadata)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"test-doc-v1::chunk-{i:03d}", (
            f"Unexpected chunk_id: {chunk.chunk_id}"
        )


def test_no_empty_chunks(metadata: DocumentMetadata) -> None:
    """No chunk should have empty or whitespace-only content."""
    chunker = RecursiveTextChunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk("A " * 500, metadata)
    for chunk in chunks:
        assert chunk.content.strip(), "Found empty chunk"


def test_overlap_preserved(metadata: DocumentMetadata) -> None:
    """
    TODO (Phase 1): verify that the last N tokens of chunk[i] appear at the
    start of chunk[i+1] when overlap > 0.
    """
    pytest.skip("Implement after RecursiveTextChunker._split_recursive() is done")


def test_single_paragraph_within_budget(metadata: DocumentMetadata) -> None:
    """Short text should produce exactly one chunk."""
    chunker = RecursiveTextChunker(chunk_size=500, overlap=50)
    text = "This is a short paragraph that fits in one chunk."
    chunks = chunker.chunk(text, metadata)
    assert len(chunks) == 1
    assert chunks[0].content == text
