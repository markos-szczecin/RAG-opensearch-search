"""
FastAPI dependency injection container.

All singletons are created once via functools.lru_cache and injected into
route handlers via FastAPI's Depends() mechanism.  This keeps routes thin and
makes every service trivially testable by overriding app.dependency_overrides.
"""

from functools import lru_cache

from opensearchpy import AsyncOpenSearch

from app.config import Settings, get_settings
from app.evaluation.repository import EvalRepository
from app.indexing.embedder.openai import OpenAIEmbedder
from app.indexing.opensearch_indexer import OpenSearchIndexer
from app.indexing.pipeline import IngestionPipeline
from app.rag.graph import build_rag_graph
from app.search.hybrid import HybridSearchService
from app.search.index_manager import IndexManager
from app.search.keyword import KeywordSearchService
from app.search.vector import VectorSearchService


@lru_cache(maxsize=1)
def get_opensearch_client() -> AsyncOpenSearch:
    settings = get_settings()
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False,
        # TODO: enable SSL + cert verification for production
        # TODO: add HTTP basic auth when DISABLE_SECURITY_PLUGIN=false
    )


@lru_cache(maxsize=1)
def get_embedder() -> OpenAIEmbedder:
    settings = get_settings()
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


@lru_cache(maxsize=1)
def get_index_manager() -> IndexManager:
    return IndexManager(
        client=get_opensearch_client(),
        index_name=get_settings().opensearch_index,
        embedding_dimensions=get_settings().embedding_dimensions,
    )


@lru_cache(maxsize=1)
def get_opensearch_indexer() -> OpenSearchIndexer:
    return OpenSearchIndexer(
        client=get_opensearch_client(),
        index_name=get_settings().opensearch_index,
    )


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        embedder=get_embedder(),
        indexer=get_opensearch_indexer(),
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_keyword_search() -> KeywordSearchService:
    return KeywordSearchService(
        client=get_opensearch_client(),
        index_name=get_settings().opensearch_index,
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_vector_search() -> VectorSearchService:
    return VectorSearchService(
        client=get_opensearch_client(),
        embedder=get_embedder(),
        index_name=get_settings().opensearch_index,
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_hybrid_search() -> HybridSearchService:
    return HybridSearchService(
        keyword_service=get_keyword_search(),
        vector_service=get_vector_search(),
        alpha=get_settings().hybrid_alpha,
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_rag_graph():  # type: ignore[return]
    """Return the compiled LangGraph RAG workflow."""
    return build_rag_graph(
        keyword_search=get_keyword_search(),
        vector_search=get_vector_search(),
        hybrid_search=get_hybrid_search(),
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_eval_repository() -> EvalRepository:
    return EvalRepository(dsn=get_settings().postgres_dsn)
