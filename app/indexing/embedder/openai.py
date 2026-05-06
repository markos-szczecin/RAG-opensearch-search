import asyncio
import hashlib

from openai import AsyncOpenAI

from app.indexing.embedder.base import Embedder

_BATCH_SIZE = 100   # OpenAI recommends ≤ 2048 strings; 100 is safe for long docs


class OpenAIEmbedder(Embedder):
    """
    Embedding implementation backed by OpenAI's text-embedding-3-small (default).

    TODO (Phase 1 — implement these before indexing real documents):
      - Batching: split texts into _BATCH_SIZE groups, call API once per batch,
        then flatten results in order.
      - Retry: wrap API calls with tenacity retry on openai.RateLimitError and
        openai.APIConnectionError with exponential back-off (max 3 attempts).
      - Caching: compute sha256(text) as cache key; store embeddings in an
        in-process dict or Redis. Skip API call on cache hit. This matters when
        re-indexing documents that haven't changed.
      - Cost tracking: accumulate usage.total_tokens across batches and expose
        as a property for the IngestionPipeline to report.
    """

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._cache: dict[str, list[float]] = {}   # content-hash → vector

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        TODO: implement batching + retry.
        Stub returns zero vectors so the pipeline compiles without API keys.
        """
        return [[0.0] * self._dimensions for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        """
        TODO: implement with retry.
        For text-embedding-3-* the query and document instructions are the same,
        but add an 'input_type' param if switching to a model that requires it.
        """
        results = await self.embed_texts([text])
        return results[0]

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
