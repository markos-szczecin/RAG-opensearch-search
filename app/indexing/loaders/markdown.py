"""
MarkdownLoader — wczytuje pliki Markdown jako sekcje gotowe do chunkowania.

Dlaczego Markdown zamiast plaintext?
--------------------------------------
Dokumenty fintech są często pisane w Markdown lub konwertowane do niego z Word/PDF.
Markdown daje nam strukturę za darmo:
  - YAML front matter: tytuł, daty ważności, poziom dostępu — metadane dokładnie tam,
    gdzie jest treść, nie w osobnym rejestrze
  - Nagłówki H1/H2: naturalne granice tematyczne, idealne do podziału na sekcje
  - Formatowanie listy, tabel: zachowane w tekście dla modelu językowego

Dlaczego podział na sekcje H2 jest lepszy niż płaski chunking?
----------------------------------------------------------------
Bez podziału: chunker dostaje cały dokument jako jeden strumień tekstu i dzieli
go mechanicznie co N tokenów. Granice chunków mogą przypadać w środku akapitu
lub między zdaniami z różnych sekcji tematycznych.

Z podziałem na H2: każda sekcja ("## Procedura autoryzacji", "## Limity przelewów")
trafia do chunkera jako osobny blok. Chunker rzadko musi dzielić dalej — sekcje
fintech mają zwykle 200-600 tokenów. Efekt: chunki naturalnie odpowiadają
jednemu zagadnieniu, co poprawia precyzję cytowań.

YAML front matter — co to jest?
---------------------------------
Front matter to blok YAML na początku pliku Markdown, otoczony ---:
```
---
title: Polityka Autoryzacji Mobilnej
doc_type: policy
valid_from: 2024-01-01
valid_to: 2025-12-31
access_level: internal
---
# Tytuł dokumentu
...treść...
```
Parsujemy go i wzbogacamy metadane dokumentu — jeśli plik ma front matter,
nie musimy podawać metadanych ręcznie przy indeksowaniu.
"""

import logging
import re
from pathlib import Path

import yaml

from app.indexing.loaders.base import DocumentLoader
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)

# Wyrażenie regularne dopasowujące YAML front matter.
# re.DOTALL: "." dopasowuje też znaki nowej linii (potrzebne dla wieloliniowego YAML).
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Dopasowuje nagłówek H1: linia zaczynająca się od "# " (dokładnie jeden #)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# Dopasowuje nagłówek H2: linia zaczynająca się od "## " (dokładnie dwa #)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


class MarkdownLoader(DocumentLoader):
    """
    Wczytuje pliki Markdown i dzieli na sekcje poziomem H2.

    Pipeline przetwarzania:
      1. Odczyt pliku (UTF-8)
      2. Wyodrębnienie i parsowanie YAML front matter
      3. Wzbogacenie metadanych z front matter (jeśli nie podano ręcznie)
      4. Ekstrakcja tytułu z H1 (jeśli brak w metadanych)
      5. Podział na sekcje po nagłówkach H2
      6. Zwrócenie listy (tekst_sekcji, metadane) gotowej do chunkowania

    Każda sekcja H2 zawiera nagłówek na początku — model językowy widzi
    kontekst sekcji nawet jeśli pobiera tylko jeden chunk z jej środka.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, path: Path, metadata: DocumentMetadata) -> list[tuple[str, DocumentMetadata]]:
        """
        Wczytuje plik Markdown i zwraca listę (tekst_sekcji, metadane).

        Metadane z front matter uzupełniają te podane przez wywołującego,
        ale nie nadpisują ich — dane podane explicite mają wyższy priorytet.
        Wyjątek: tytuł z H1 jest ustawiany tylko gdy metadata.title jest pusty.

        Args:
            path:     Ścieżka do pliku .md lub .markdown.
            metadata: Metadane dokumentu podane przez wywołującego.
                      Mogą być częściowe — loader uzupełni brakujące pola z front matter.

        Returns:
            Lista krotek (tekst, metadane). Zazwyczaj kilka pozycji (jedna per sekcja H2).
            Jeden element gdy brak sekcji H2 lub dokument jednoakapitowy.
        """
        raw = path.read_text(encoding="utf-8")

        # ------------------------------------------------------------------
        # Krok 1: Wyodrębnij i sparsuj YAML front matter
        # ------------------------------------------------------------------
        enriched_meta = metadata
        body = raw

        fm_match = _FRONT_MATTER_RE.match(raw)
        if fm_match:
            try:
                fm_data = yaml.safe_load(fm_match.group(1)) or {}
                # Odetnij front matter od treści dokumentu
                body = raw[fm_match.end():]

                # Wzbogać metadane polami z front matter, ale tylko gdy wywołujący
                # nie podał już wartości. Dane z kodu (indeksowanie z zewnętrznego
                # rejestru) mają wyższy priorytet niż dane z pliku.
                updates: dict = {}
                if not metadata.title and fm_data.get("title"):
                    updates["title"] = str(fm_data["title"])
                if metadata.doc_type == "unknown" and fm_data.get("doc_type"):
                    updates["doc_type"] = str(fm_data["doc_type"])
                if metadata.access_level == "public" and fm_data.get("access_level"):
                    updates["access_level"] = str(fm_data["access_level"])
                # Daty ważności z front matter — ważne dla freshness filtering
                if not metadata.valid_from and fm_data.get("valid_from"):
                    updates["valid_from"] = fm_data["valid_from"]
                if not metadata.valid_to and fm_data.get("valid_to"):
                    updates["valid_to"] = fm_data["valid_to"]

                if updates:
                    # model_copy tworzy nową instancję z zaktualizowanymi polami.
                    # Modele Pydantic są niemutowalne — nie można bezpośrednio
                    # przypisać atrybutu. model_copy to oficjalny sposób "zmiany".
                    enriched_meta = metadata.model_copy(update=updates)

            except yaml.YAMLError as exc:
                # Uszkodzony front matter: logujemy ostrzeżenie i traktujemy
                # cały plik jako treść (bez wzbogacania metadanych z pliku).
                logger.warning("Nieprawidłowy front matter YAML w %s: %s", path, exc)
                body = raw

        # ------------------------------------------------------------------
        # Krok 2: Wyodrębnij tytuł z pierwszego nagłówka H1
        # ------------------------------------------------------------------
        if not enriched_meta.title:
            h1_match = _H1_RE.search(body)
            if h1_match:
                enriched_meta = enriched_meta.model_copy(
                    update={"title": h1_match.group(1).strip()}
                )

        # ------------------------------------------------------------------
        # Krok 3: Podziel na sekcje po nagłówkach H2
        # ------------------------------------------------------------------
        # re.split z grupą przechwytującą zwraca przeplatane: [tekst, nagłówek, tekst, ...]
        # Przykład dla "intro\n## Sekcja 1\ntreść\n## Sekcja 2\ntreść":
        #   ["intro\n", "Sekcja 1", "\ntreść\n", "Sekcja 2", "\ntreść"]
        sections = _H2_RE.split(body)

        if len(sections) <= 1:
            # Brak nagłówków H2 → cały dokument jako jedna sekcja
            clean = body.strip()
            return [(clean, enriched_meta)] if clean else []

        results: list[tuple[str, DocumentMetadata]] = []

        # Tekst przed pierwszym H2 (wstęp, definicje, tytuł dokumentu)
        intro = sections[0].strip()
        if intro:
            results.append((intro, enriched_meta))

        # Sekcje H2: pairs (nagłówek, treść) na indeksach [1,2], [3,4], [5,6]...
        for i in range(1, len(sections), 2):
            heading = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            if not content:
                continue

            # Dołączamy nagłówek na początku treści sekcji.
            # Dzięki temu każdy chunk "wie" do jakiej sekcji należy —
            # kluczowe gdy chunk jest wyciągany z kontekstu całego dokumentu
            # i wysyłany do modelu językowego jako izolowany fragment.
            section_text = f"## {heading}\n\n{content}"
            results.append((section_text, enriched_meta))

        # Jeśli nic nie pasuje (np. dokument tylko z H2 bez treści), zwróć cały body
        return results if results else [(body.strip(), enriched_meta)]
