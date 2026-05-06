from opensearchpy import AsyncOpenSearch

from app.indexing.opensearch_indexer import INDEX_MAPPING


class IndexManager:
    """
    Manages the OpenSearch index lifecycle: creation, alias management, and
    zero-downtime reindexing.

    Alias strategy:
      - Active index name: fintech_kb_v{version} (e.g. fintech_kb_v1)
      - Alias:            fintech_kb  (all services read/write via alias)
      - Reindex flow:     write to fintech_kb_v2 → test → atomically switch
                          alias → delete fintech_kb_v1

    This allows schema changes (new fields, different analyzers) without
    downtime or a full re-crawl of the knowledge base.

    TODO (Phase 1):
      - Implement create_index_if_not_exists() using the INDEX_MAPPING constant.
      - Implement create_alias() that points the alias to the versioned index.
      - TODO (Phase 4 advanced):
          - Implement reindex() that creates a new versioned index, bulk-copies
            documents via OpenSearch _reindex API, runs smoke-test queries,
            then atomically switches the alias with a single _aliases call.
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        index_name: str,
        embedding_dimensions: int = 1536,
    ) -> None:
        self._client = client
        self._alias = index_name
        self._versioned = f"{index_name}_v1"
        self._embedding_dimensions = embedding_dimensions

    async def create_index_if_not_exists(self) -> bool:
        """
        Create the versioned index + alias if they don't already exist.
        Returns True if the index was created, False if it already existed.

        TODO: patch INDEX_MAPPING's knn_vector dimension from self._embedding_dimensions.
        """
        exists = await self._client.indices.exists(index=self._versioned)
        if exists:
            return False

        # TODO: create index with INDEX_MAPPING
        # await self._client.indices.create(index=self._versioned, body=INDEX_MAPPING)
        # await self.create_alias()
        raise NotImplementedError("IndexManager.create_index_if_not_exists() — implement in Phase 1")

    async def create_alias(self) -> None:
        """Point self._alias → self._versioned index."""
        # TODO: await self._client.indices.put_alias(index=self._versioned, name=self._alias)
        raise NotImplementedError

    async def get_current_version(self) -> str | None:
        """Return the versioned index name that the alias currently points to."""
        # TODO: await self._client.indices.get_alias(name=self._alias)
        raise NotImplementedError
