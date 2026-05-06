from pathlib import Path

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata


class CSVLoader(DocumentLoader):
    """
    Loads CSV files (e.g. account_limits.csv) as structured text.

    Strategy: convert each row into a natural-language sentence so that
    semantic search can find relevant rows.

    Example row:  account_type=premium, daily_limit_eur=50000
    Converted to: "Premium accounts have a daily transfer limit of 50,000 EUR."

    TODO (Phase 1):
      - Use csv.DictReader to iterate rows.
      - Apply a column-aware template to produce readable sentences.
      - Allow callers to pass a `row_template` callable for custom formatting.
      - Group rows into a single chunk or one chunk per row (configurable).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        # TODO: read CSV rows, convert to natural-language text
        raise NotImplementedError("CSVLoader.load() — implement in Phase 1")
