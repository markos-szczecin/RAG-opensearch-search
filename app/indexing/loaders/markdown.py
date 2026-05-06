from pathlib import Path

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata


class MarkdownLoader(DocumentLoader):
    """
    Loads plain Markdown files.

    TODO (Phase 1):
      - Strip YAML front matter (--- ... ---) and parse it into metadata fields.
      - Extract the first H1 heading as the document title if metadata.title is empty.
      - Optionally split on H2 headings to produce section-level tuples,
        enabling finer-grained citations like "Section: Transfer Limits".
      - Strip HTML comments and link syntax that would confuse the chunker.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        raw = path.read_text(encoding="utf-8")

        # TODO: parse front matter
        # TODO: extract title from first H1 if metadata.title is blank
        # TODO: split into sections if desired

        return [(raw, metadata)]
