from datetime import date

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata attached to every chunk. Drives filtering, RBAC, citations, and freshness."""

    doc_id: str = Field(description="Unique document identifier, e.g. 'mobile-auth-policy-v3'")
    title: str
    doc_type: str = Field(description="policy | faq | procedure | compliance | developer")
    department: str = Field(description="e.g. compliance, product, engineering")
    language: str = Field(default="en")
    access_level: str = Field(description="public | internal | confidential")
    status: str = Field(description="draft | approved | archived")
    valid_from: date
    valid_to: date | None = None
    source_path: str = Field(description="Relative path to original document")
    version: int = Field(default=1, ge=1)

    @field_validator("access_level")
    @classmethod
    def validate_access_level(cls, v: str) -> str:
        allowed = {"public", "internal", "confidential"}
        if v not in allowed:
            raise ValueError(f"access_level must be one of {allowed}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"draft", "approved", "archived"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


class Chunk(BaseModel):
    """
    A single indexable unit. Stored as one OpenSearch document.
    chunk_id format: "{doc_id}::chunk-{n:03d}"
    """

    chunk_id: str
    doc_id: str
    content: str
    content_vector: list[float] | None = None   # populated after embedding
    metadata: DocumentMetadata

    @property
    def to_opensearch_doc(self) -> dict:
        """
        Serialize to the flat dict expected by OpenSearch bulk indexing.

        TODO: ensure date fields are ISO-formatted strings.
        TODO: exclude content_vector if None (don't overwrite existing vectors on partial update).
        """
        doc = self.metadata.model_dump()
        doc["chunk_id"] = self.chunk_id
        doc["doc_id"] = self.doc_id
        doc["content"] = self.content
        doc["valid_from"] = str(self.metadata.valid_from)
        doc["valid_to"] = str(self.metadata.valid_to) if self.metadata.valid_to else None
        if self.content_vector is not None:
            doc["content_vector"] = self.content_vector
        return doc


class IngestionResult(BaseModel):
    """Returned by IngestionPipeline after processing one file."""

    doc_id: str
    n_chunks: int
    n_indexed: int
    n_failed: int
    duration_seconds: float
    tokens_used: int = 0    # embedding API tokens consumed
