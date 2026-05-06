from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract embedding interface.

    Keeps search services and the ingestion pipeline decoupled from any
    specific embedding provider. Swap OpenAIEmbedder for a
    SentenceTransformerEmbedder without touching callers.
    """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns one vector per input string.
        Implementations must preserve ordering.
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query. May use a different instruction prefix
        than embed_texts (important for asymmetric models).
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensionality. Must match the OpenSearch mapping."""
        ...
