from pathlib import Path

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata


class PDFLoader(DocumentLoader):
    """
    Loads PDF files using pypdf.

    TODO (Phase 1):
      - Use pypdf.PdfReader to extract text page by page.
      - Concatenate pages with a page-break sentinel ("\n\n--- page {n} ---\n\n")
        so the chunker can reference page numbers in citations.
      - Handle encrypted PDFs gracefully (log warning, skip).
      - Strip headers/footers that repeat on every page (e.g. page numbers).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        # TODO: import pypdf here to avoid loading it when not needed
        # from pypdf import PdfReader
        # reader = PdfReader(path)
        # pages = [page.extract_text() or "" for page in reader.pages]
        # text = "\n\n".join(pages)
        raise NotImplementedError("PDFLoader.load() — implement in Phase 1")
