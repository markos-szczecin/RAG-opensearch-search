from abc import ABC, abstractmethod
from pathlib import Path

from app.models.document import DocumentMetadata


class DocumentLoader(ABC):
    """
    Abstract base for all document loaders.

    Each concrete loader handles one file format (Markdown, PDF, CSV, …).
    The Open/Closed Principle: add new formats by subclassing, never by
    modifying existing loaders.

    Returns a list of (raw_text, metadata) tuples — one per logical section
    of the document.  Most loaders return a single tuple; CSV loaders may
    return one tuple per row or per grouped section.
    """

    @abstractmethod
    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        """
        Parse the file at `path` and return raw text sections with metadata.

        Args:
            path: Absolute path to the source file.
            metadata: Pre-populated metadata for this document; loaders may
                      enrich it (e.g. extract title from Markdown heading).

        Returns:
            List of (text, metadata) pairs ready for chunking.
        """
        ...

    def supports(self, path: Path) -> bool:
        """Return True if this loader can handle the given file extension."""
        return path.suffix.lower() in self.supported_extensions

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this loader handles, e.g. ['.md', '.markdown']."""
        ...
