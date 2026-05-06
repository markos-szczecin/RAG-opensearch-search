from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- OpenSearch ----
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "fintech_kb"

    # ---- PostgreSQL ----
    postgres_dsn: str = "postgresql+asyncpg://rag:secret@localhost:5432/rag_eval"

    # ---- OpenAI (embeddings) ----
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ---- Anthropic (LLM) ----
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # ---- RAG token budget ----
    max_chunks: int = 5
    max_context_tokens: int = 2500
    max_tokens_per_chunk: int = 400
    max_answer_tokens: int = 600

    # ---- Retrieval ----
    default_retrieval_mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    hybrid_alpha: float = 0.5       # weight for BM25; (1 - alpha) for vector
    retrieve_top_k: int = 20        # candidates before reranking
    rerank_top_k: int = 5           # chunks sent to LLM after reranking

    # ---- Role → access level mapping ----
    # Extend this dict when adding new roles.
    # Each role is granted access to its level and all less-sensitive levels.
    role_access_levels: dict[str, list[str]] = {
        "customer": ["public"],
        "support_agent": ["public", "internal"],
        "compliance_officer": ["public", "internal", "confidential"],
        "developer": ["public", "internal"],
        "admin": ["public", "internal", "confidential"],
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
