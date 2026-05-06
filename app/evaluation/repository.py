"""
PostgreSQL-backed repository for evaluation and feedback logs.

Uses SQLAlchemy async ORM so all DB calls are non-blocking.

Tables:
  search_log  — one row per search request (latency, mode, token count)
  feedback    — user-submitted ratings + comments
  eval_run    — one row per golden-query evaluation result

TODO (Phase 4):
  - Implement create_tables() using SQLAlchemy metadata.create_all().
  - Implement log_search(), log_feedback(), log_eval_result() with real DB writes.
  - Add get_feedback_stats() for the admin dashboard.
  - Add Alembic migrations so schema changes don't require DROP TABLE.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.models.feedback import FeedbackRequest


class Base(DeclarativeBase):
    pass


class SearchLog(Base):
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
    __tablename__ = "eval_run"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text)
    retrieval_mode = Column(String(20))
    expected_doc_ids = Column(Text)    # JSON array stored as text
    retrieved_doc_ids = Column(Text)   # JSON array stored as text
    precision_at_5 = Column(Float)
    recall_at_5 = Column(Float)
    mrr = Column(Float)
    latency_ms = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class EvalRepository:
    """
    Async repository for all DB write/read operations.

    Instantiated once in dependencies.py and injected via FastAPI Depends().
    """

    def __init__(self, dsn: str) -> None:
        self._engine: AsyncEngine = create_async_engine(dsn, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """Run DDL to create all tables if they don't exist."""
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
        Persist a search log entry.

        TODO: implement async session write.
        Returns the generated log ID.
        """
        raise NotImplementedError("EvalRepository.log_search() — implement in Phase 4")

    async def log_feedback(self, request: FeedbackRequest) -> str:
        """
        TODO: persist feedback entry and return its ID.
        """
        raise NotImplementedError

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
        """TODO: persist evaluation run result."""
        raise NotImplementedError

    async def close(self) -> None:
        await self._engine.dispose()
