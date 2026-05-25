"""
PDFLoader — wyodrębnia tekst z plików PDF strona po stronie.

Wyzwania związane z PDF w systemach RAG
-----------------------------------------
PDF jest formatem prezentacji, nie struktury. W odróżnieniu od Markdown nie ma
w nim pojęcia "akapitu" czy "sekcji" — jest tylko rozmieszczenie znaków na stronie.

Typowe problemy:
  1. Nagłówki/stopki: numer strony i nazwa dokumentu powtarzają się na każdej stronie
     i stają się szumem w indeksie wektorowym. Trudne do automatycznego usunięcia.
  2. Tabele: wyodrębniony tekst tabeli to chaos — komórki są scalane bez separatorów.
     pypdf nie obsługuje tabel dobrze (potrzeba camelot lub pdfplumber).
  3. Wielokolumnowy układ: tekst jest wyodrębniony w kolejności kolumn, ale
     pypdf czasem myli kolumny → niespójna kolejność słów.
  4. Zaszyfrowane PDF: wymagają hasła; pomijamy je z ostrzeżeniem.
  5. Skany (PDF-y z obrazkami): pypdf nie wyodrębni tekstu — potrzeba OCR (Tesseract).

Podejście w tym loaderze
--------------------------
Proste wyodrębnienie tekstu przez pypdf, strona po stronie.
Każda strona jest oznaczona sentynenlem "--- page N ---" który:
  - Pozwala chunkerowi zachować informację o numerze strony
  - Umożliwia cytowanie "Dokument X, strona 3" zamiast ogólnie "Dokument X"
  - Daje naturalną granicę dla dużych dokumentów

Dla wysokiej jakości ekstrakcji tabel i złożonego układu rozważ pdfplumber
(dokładniejszy, ale wolniejszy) lub nanalytics/zerox (LLM-based, najdokładniejszy).
"""

import logging
from pathlib import Path

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """
    Wczytuje pliki PDF jako tekst z sentynelami stron.

    Lazy import pypdf: biblioteka jest importowana dopiero przy pierwszym wywołaniu
    load(). Dzięki temu startup aplikacji jest szybszy i nie wymaga pypdf
    gdy nie ma żadnych dokumentów PDF do zaindeksowania.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        """
        Wyodrębnia tekst z PDF strona po stronie.

        Strony są łączone w jeden dokument ze sentynelami "--- page N ---".
        Chunker podzieli ten tekst na fragmenty — sentynele pomagają zachować
        informację o lokalizacji w dokumencie źródłowym.

        Args:
            path:     Ścieżka do pliku PDF.
            metadata: Metadane dokumentu (tytuł, poziom dostępu itp.).

        Returns:
            Pojedynczy element listy z pełnym tekstem dokumentu i metadanymi.
            Pusta lista jeśli PDF jest zaszyfrowany lub nie zawiera tekstu.

        Raises:
            ValueError: Gdy pliku nie można otworzyć jako PDF.
        """
        # Lazy import — ładuj pypdf tylko gdy potrzebny
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf jest wymagany do wczytywania plików PDF. "
                "Zainstaluj: pip install pypdf"
            ) from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise ValueError(f"Nie można otworzyć PDF {path}: {exc}") from exc

        # Zaszyfrowane PDF: jeśli nie mamy hasła, pomijamy z ostrzeżeniem.
        # Rzucanie wyjątku przerwałoby całą ingestion directory — lepiej pominąć.
        if reader.is_encrypted:
            logger.warning(
                "Pomijam zaszyfrowany PDF: %s. "
                "Odchyfruj plik i zindeksuj ponownie.",
                path,
            )
            return []

        pages: list[str] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                # Sentinel strony: pozwala na cytowanie "Dokument X, strona N"
                # i daje chunkerowi naturalną granicę między stronami.
                # Podwójny \n\n na końcu = granica akapitu dla RecursiveTextChunker.
                pages.append(f"--- page {page_num} ---\n{text}")

        if not pages:
            logger.warning(
                "PDF %s nie zawiera wyodrębnialnego tekstu. "
                "Może to być skan — rozważ OCR (Tesseract).",
                path,
            )
            return []

        # Łączymy wszystkie strony w jeden ciąg.
        # Chunker podzieli go na fragmenty respektując sentynele stron.
        full_text = "\n\n".join(pages)
        return [(full_text, metadata)]
