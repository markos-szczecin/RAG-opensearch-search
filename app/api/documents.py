from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_ingestion_pipeline
from app.indexing.pipeline import IngestionPipeline
from app.models.document import DocumentMetadata, IngestionResult

router = APIRouter()


class IndexRequest(BaseModel):
    """
    Request body for document ingestion.

    source_path: path relative to the /docs_sample mount (or absolute inside container).
    metadata: document-level metadata applied to all chunks from this file.

    TODO: add a batch variant that accepts a list of IndexRequest objects so
    operators can trigger a full directory re-index in a single API call.
    """

    source_path: str
    metadata: DocumentMetadata


@router.post("/index", response_model=IngestionResult, status_code=202)
async def index_document(request: IndexRequest,pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)) -> IngestionResult:
    """
    Ingest a single document into OpenSearch.

    Returns 202 Accepted with stats: n_chunks, n_indexed, duration, tokens_used.

    TODO (Phase 1):
      - Run pipeline.ingest_file() and return the result.
      - Add background task support (FastAPI BackgroundTasks) so large
        documents don't block the HTTP response.
    """
    path = Path(request.source_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.absolute()}")

    try:
        return await pipeline.ingest_file(path, request.metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # raise HTTPException(status_code=501, detail="Indexing not yet implemented — Phase 1")
