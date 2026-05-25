"""
EvalRepository — PostgreSQL-backed repozytorium do logowania zapytań, feedbacku i ewaluacji.

Dlaczego potrzebujemy osobnej bazy danych skoro OpenSearch już wszystko indeksuje?
-----------------------------------------------------------------------------------
OpenSearch to silnik wyszukiwania — optymalizowany do fast retrieval, nie do
zapisu transakcyjnego i analityki. PostgreSQL to relacyjna baza danych — idealna do:
  - Logowania każdego zapytania z latency, tokenami, rolą (tabela search_log)
  - Zbierania feedbacku użytkowników (rating 1-5 + komentarz) w tabeli feedback
  - Przechowywania wyników ewaluacji złotego zbioru zapytań (tabela eval_run)

Te dane pozwalają odpowiedzieć na pytania:
  - "Które zapytania mają najniższy rating?" → debugowanie jakości
  - "Czy hybrid search jest szybszy niż keyword?" → porównanie trybów
  - "Jaki jest trend latency w czasie?" → monitoring regresu

Wzorzec Repository
-------------------
Repository enkapsuluje wszystkie operacje na bazie danych — reszta aplikacji
nie wie nic o SQLAlchemy ani SQL. Korzyści:
  1. Łatwa wymiana implementacji (np. PostgreSQL → DynamoDB) bez zmiany kodu
  2. Testowanie z MockRepository zamiast prawdziwej bazy
  3. Jeden punkt do dodania logowania, metryk, transakcji

Async SQLAlchemy
-----------------
Używamy AsyncSession (z asyncpg jako driver) żeby nie blokować event loop FastAPI
podczas operacji na bazie. Synchroniczny SQLAlchemy zablokuje wszystkie inne
requesty w czasie oczekiwania na DB. Dla endpoint'u POST /ask który może trwać
1-2 sekundy, to krytyczne.

Schemat tabel:
  search_log: każde zapytanie /ask lub /search z metadanymi
  feedback:   ocena (1-5) + komentarz użytkownika po odpowiedzi
  eval_run:   wyniki ewaluacji P@k, R@k, MRR dla golden queries
"""

import json
import time
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.models.feedback import FeedbackRequest


class Base(DeclarativeBase):
    """Bazowa klasa dla wszystkich modeli ORM SQLAlchemy."""
    pass


class SearchLog(Base):
    """
    Jeden rekord per wywołanie /ask lub /search.

    Przechowuje kontekst potrzebny do debugowania jakości odpowiedzi:
    - Zapytanie (do analizy wzorców użycia)
    - Tryb retrieval (który tryb użyto)
    - Latency (czy system działa wystarczająco szybko?)
    - Tokeny (ile kosztuje każde zapytanie?)
    - Rola użytkownika (czy różne role mają różne zachowania?)
    """
    __tablename__ = "search_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    retrieval_mode = Column(String(20))
    latency_ms = Column(Float)
    context_tokens = Column(Integer)
    answer_tokens = Column(Integer)
    n_results = Column(Integer)
    user_role = Column(String(50))
    recorded_at = Column(DateTime, default=datetime.utcnow)


class FeedbackLog(Base):
    """
    Ocena i komentarz od użytkownika po otrzymaniu odpowiedzi.

    Rating 1-5 gdzie:
      1 = Odpowiedź całkowicie błędna lub nieistotna
      3 = Odpowiedź częściowo poprawna
      5 = Odpowiedź dokładna i pomocna

    Feedback jest kluczowy dla uczenia systemu — identyfikuje
    zapytania gdzie system zawodzi i gdzie radzi sobie dobrze.
    """
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text)
    answer = Column(Text)
    rating = Column(Integer)
    comment = Column(Text, nullable=True)
    user_role = Column(String(50))
    retrieval_mode = Column(String(20), nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class EvalRun(Base):
    """
    Wynik jednego zapytania z golden query set.

    Jeden rekord per (query, retrieval_mode) — pozwala porównywać
    tryby wyszukiwania na tym samym zestawie testowym.

    expected_doc_ids i retrieved_doc_ids przechowywane jako JSON string
    bo SQLAlchemy bazowe nie obsługuje natywnie list (bez Postgres-specific JSON column).
    """
    __tablename__ = "eval_run"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text)
    retrieval_mode = Column(String(20))
    expected_doc_ids = Column(Text)    # JSON array: ["doc-001", "doc-002"]
    retrieved_doc_ids = Column(Text)   # JSON array: ["doc-001", "doc-003"]
    precision_at_5 = Column(Float)
    recall_at_5 = Column(Float)
    mrr = Column(Float)
    latency_ms = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class EvalRepository:
    """
    Async repozytorium dla wszystkich operacji DB: zapis logów i odczyt statystyk.

    Wzorzec session factory:
      Zamiast trzymać jedno globalne połączenie, tworzymy nową sesję dla każdej
      operacji przez session_factory() jako async context manager.
      Zalety: automatyczne zamknięcie sesji, thread-safety, connection pooling.

    Wzorzec "session per unit of work":
      Każda metoda otwiera sesję, zapisuje jeden rekord i commituje.
      To odpowiednik jednej transakcji SQL.
      Wada: brak transakcji między metodami (jeśli log_search i log_feedback
      muszą być w jednej transakcji, trzeba przekazać sesję między metodami).
    """

    def __init__(self, dsn: str) -> None:
        """
        Args:
            dsn: Connection string dla asyncpg, np.:
                 "postgresql+asyncpg://user:password@host:5432/dbname"
                 Protokół "postgresql+asyncpg://" jest wymagany dla async driver.
                 SQLAlchemy samodzielnie wybiera asyncpg jako driver.
        """
        self._engine: AsyncEngine = create_async_engine(
            dsn,
            # echo=False: nie loguj każdego SQL statement (włącz echo=True do debugowania)
            echo=False,
            # pool_size: liczba stałych połączeń w puli
            pool_size=5,
            # max_overflow: dodatkowe połączenia gdy pula jest pełna
            max_overflow=10,
        )
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            # expire_on_commit=False: pozwala na dostęp do atrybutów po commit
            # bez lazy-loadingu (potrzebne dla async — nie można lazy-loadować po zamknięciu sesji)
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        """
        Tworzy wszystkie tabele jeśli nie istnieją (idempotentne).

        run_sync() wymaga synchronicznej funkcji — metadata.create_all jest
        synchroniczne, ale uruchamiamy ją w async context przez run_sync.
        Wywołane przy starcie aplikacji (lifespan w main.py).
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def log_search(
        self,
        query: str,
        retrieval_mode: str,
        latency_ms: float,
        context_tokens: int,
        answer_tokens: int,
        n_results: int,
        user_role: str,
    ) -> str:
        """
        Zapisuje log jednego żądania wyszukiwania/answering.

        Returns:
            UUID nowo zapisanego rekordu (do dalszego śledzenia feedbacku).
        """
        entry = SearchLog(
            query=query,
            retrieval_mode=retrieval_mode,
            latency_ms=latency_ms,
            context_tokens=context_tokens,
            answer_tokens=answer_tokens,
            n_results=n_results,
            user_role=user_role,
        )
        # "async with session_factory() as session" otwiera połączenie z puli,
        # tworzy sesję i automatycznie rollback'uje przy wyjątku.
        async with self._session_factory() as session:
            session.add(entry)
            await session.commit()
        return str(entry.id)

    async def log_feedback(self, request: FeedbackRequest) -> str:
        """
        Zapisuje ocenę użytkownika dla otrzymanej odpowiedzi.

        FeedbackRequest zawiera zarówno treść odpowiedzi jak i ocenę,
        co pozwala na późniejsze powiązanie feedbacku z konkretną wersją systemu.

        Returns:
            UUID nowo zapisanego rekordu feedbacku.
        """
        entry = FeedbackLog(
            query=request.query,
            answer=request.answer,
            rating=request.rating,
            comment=request.comment,
            user_role=request.user_role,
            retrieval_mode=request.retrieval_mode,
        )
        async with self._session_factory() as session:
            session.add(entry)
            await session.commit()
        return str(entry.id)

    async def log_eval_result(
        self,
        query: str,
        retrieval_mode: str,
        expected_doc_ids: list[str],
        retrieved_doc_ids: list[str],
        precision_at_5: float,
        recall_at_5: float,
        mrr: float,
        latency_ms: float,
    ) -> None:
        """
        Zapisuje wynik ewaluacji dla jednego golden query.

        Listy doc_id są serializowane do JSON string bo używamy prostych typów
        SQLAlchemy bez Postgres-specific JSON column (dla przenośności między DB).
        Przy odczycie: json.loads(eval_run.expected_doc_ids) przywraca listę.
        """
        entry = EvalRun(
            query=query,
            retrieval_mode=retrieval_mode,
            expected_doc_ids=json.dumps(expected_doc_ids),
            retrieved_doc_ids=json.dumps(retrieved_doc_ids),
            precision_at_5=precision_at_5,
            recall_at_5=recall_at_5,
            mrr=mrr,
            latency_ms=latency_ms,
        )
        async with self._session_factory() as session:
            session.add(entry)
            await session.commit()

    async def ping(self) -> float:
        """
        Sprawdza dostępność bazy i mierzy latency.

        SELECT 1 to minimalne zapytanie SQL — nie dotyka żadnych danych,
        tylko sprawdza czy połączenie działa. Zwraca latency w ms.

        Używane przez health endpoint (/health) do raportowania stanu bazy.

        Returns:
            Latency w milisekundach dla połączenia z bazą.

        Raises:
            Exception: gdy połączenie z bazą jest niedostępne.
        """
        start = time.monotonic()
        async with self._session_factory() as session:
            # text() opakowuje surowy SQL — wymagane przez SQLAlchemy 2.x
            # dla bezpieczeństwa (zapobiega accidental SQL injection przez
            # przekazanie niesanityzowanego stringa)
            await session.execute(text("SELECT 1"))
        return (time.monotonic() - start) * 1000

    async def close(self) -> None:
        """Zamknij connection pool — wywoływane przy shutdown aplikacji."""
        await self._engine.dispose()
