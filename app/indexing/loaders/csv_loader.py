"""
CSVLoader — konwertuje pliki CSV na tekst naturalny gotowy do wyszukiwania.

Problem: CSV jest formatem tabelarycznym, a wyszukiwanie semantyczne działa
na zdaniach naturalnego języka. Prosta konkatenacja kolumn ("premium 50000 EUR")
jest słabym sygnałem dla modelu embeddings — brakuje kontekstu gramatycznego.

Podejście: przekształcamy każdy wiersz CSV w pełne zdanie opisujące jego treść.

Przed transformacją (wiersz CSV):
  account_type=premium, daily_limit_eur=50000, foreign_transfer_allowed=true

Po transformacji (zdanie NL):
  "Premium accounts have a daily transfer limit of 50,000 EUR. Foreign transfers are
   allowed with a daily limit of 25,000 EUR. ATM daily limit: 5,000 EUR."

Dlaczego to działa lepiej dla wyszukiwania semantycznego?
----------------------------------------------------------
Model embeddings był trenowany na tekście naturalnym, nie na surowych wartościach tabel.
Zdanie "maximum single transfer limit" semantycznie pasuje do zapytania "jak duży
przelew mogę zrobić?" — klucz "single_transfer_limit_eur=10000" już nie.

Ograniczenia tej implementacji
--------------------------------
  - Schemat tabeli musi być znany z góry dla dobrego formatowania.
    Nieznany schemat dostaje generic fallback ("klucz: wartość, ...").
  - Dla tabel z > 1000 wierszy rozważ jedno zdanie per wiersz zamiast łączenia
    (każdy wiersz jako osobny "dokument" do wyszukiwania).
  - Tabele relacyjne (join'ów) wymagają przetworzenia przed załadowaniem.
"""

import csv
import logging
from pathlib import Path

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)


class CSVLoader(DocumentLoader):
    """
    Wczytuje pliki CSV i konwertuje wiersze na zdania naturalnego języka.

    Obsługuje dwa tryby:
      1. Znany schemat (account_limits): dedykowany szablon zdania z kontekstem
      2. Nieznany schemat: generic fallback "klucz: wartość" dla każdej kolumny

    Wszystkie wiersze są łączone w jeden dokument, który chunker podzieli
    na fragmenty. Dla małych tabel (< 50 wierszy) jest to optymalne —
    chunker rzadko musi dzielić i każdy chunk zawiera kilka powiązanych wierszy.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        """
        Wczytuje CSV i konwertuje na tekst naturalny.

        Każdy wiersz jest konwertowany osobno, a zdania łączone dwoma newline'ami
        (granica akapitu dla RecursiveTextChunker → każde zdanie może być osobnym
        chunkiem przy małym chunk_size).

        Args:
            path:     Ścieżka do pliku CSV (UTF-8 lub UTF-8-BOM).
            metadata: Metadane dokumentu.

        Returns:
            Lista z jednym elementem (cały tekst + metadane) lub pusta lista.
        """
        rows: list[str] = []

        try:
            # newline="" wymagane przez dokumentację csv.reader — nie dodawaj
            # własnych \n; csv.reader sam obsługuje różne zakończenia linii
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):  # start=2: nagłówek = linia 1
                    try:
                        sentence = self._row_to_sentence(row)
                        if sentence:
                            rows.append(sentence)
                    except Exception as exc:
                        logger.warning(
                            "Błąd konwersji wiersza %d w %s: %s",
                            row_num, path, exc,
                        )
                        # Pomiń zepsuty wiersz, kontynuuj z resztą

        except UnicodeDecodeError:
            # Fallback do latin-1 dla starszych plików CSV z polskich systemów bankowych
            logger.warning(
                "UTF-8 nie działa dla %s — próbuję latin-1 (CP1250)", path
            )
            with path.open(encoding="latin-1", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sentence = self._row_to_sentence(row)
                    if sentence:
                        rows.append(sentence)

        if not rows:
            logger.warning("CSV %s nie zawiera żadnych danych po konwersji", path)
            return []

        # Łączymy wiersze podwójnym newline — naturalna granica akapitu
        combined = "\n\n".join(rows)
        return [(combined, metadata)]

    def _row_to_sentence(self, row: dict) -> str:
        """
        Konwertuje jeden wiersz CSV na zdanie naturalnego języka.

        Sprawdza czy wiersz pasuje do znanych schematów (account_limits).
        Dla nieznanego schematu używa generic fallback.

        Dlaczego sprawdzamy klucze a nie nazwę pliku?
        -----------------------------------------------
        Loader może być wywołany dla różnych plików z tym samym schematem
        (np. account_limits_2024.csv, account_limits_2025.csv). Sprawdzanie
        kluczy kolumn jest bardziej odporne niż dopasowanie nazwy pliku.
        """
        if self._is_account_limits_schema(row):
            return self._format_account_limits(row)

        # Generic fallback — dla nieznanych schematów
        # Pomijamy puste wartości aby nie tworzyć zdań z dziurami
        parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
        return ". ".join(parts) + "." if parts else ""

    @staticmethod
    def _is_account_limits_schema(row: dict) -> bool:
        """Sprawdza czy wiersz ma schemat tabeli account_limits."""
        required_cols = {"account_type", "daily_limit_eur"}
        return required_cols.issubset(row.keys())

    @staticmethod
    def _format_account_limits(row: dict) -> str:
        """
        Formatuje wiersz tabeli limitów kont jako czytelne zdanie angielskie.

        Schemat account_limits.csv:
          account_type, daily_limit_eur, monthly_limit_eur, single_transfer_limit_eur,
          foreign_transfer_allowed, foreign_daily_limit_eur, atm_daily_limit_eur,
          instant_payment_limit_eur

        Zdanie jest zaprojektowane tak, by odpowiadać na typowe pytania użytkowników:
          "jaki mam limit dzienny?", "czy mogę robić przelewy za granicę?",
          "ile mogę wypłacić w bankomacie?"
        """
        account = row.get("account_type", "").replace("_", " ").title()
        daily = row.get("daily_limit_eur", "N/A")
        monthly = row.get("monthly_limit_eur", "N/A")
        single = row.get("single_transfer_limit_eur", "N/A")
        foreign_raw = row.get("foreign_transfer_allowed", "false").lower()
        foreign_allowed = foreign_raw in ("true", "yes", "1", "tak")
        foreign_daily = row.get("foreign_daily_limit_eur", "0")
        atm = row.get("atm_daily_limit_eur", "N/A")
        instant = row.get("instant_payment_limit_eur", "N/A")

        foreign_part = (
            f"Foreign transfers are allowed with a daily limit of {foreign_daily} EUR."
            if foreign_allowed
            else "Foreign transfers are not allowed."
        )

        return (
            f"{account} accounts have a daily transfer limit of {daily} EUR, "
            f"a monthly limit of {monthly} EUR, "
            f"and a single transfer limit of {single} EUR. "
            f"{foreign_part} "
            f"ATM daily withdrawal limit: {atm} EUR. "
            f"Instant payment limit: {instant} EUR."
        )
