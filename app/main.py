from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.config import get_settings
from app.dependencies import get_eval_repository, get_opensearch_client, get_index_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup / shutdown lifecycle.

    TODO (Phase 1): on startup
      - call IndexManager.create_index_if_not_exists() to ensure the OS index + mapping exist
      - call EvalRepository.create_tables() to run SQLAlchemy DDL
      - optionally warm the embedder by sending a dummy sentence

    TODO (Phase 4): on shutdown
      - close the AsyncOpenSearch client connection pool
      - close the SQLAlchemy async engine
    """
    settings = get_settings()
    os_client = get_opensearch_client()
    index_manager = get_index_manager()
    eval_repo = get_eval_repository()

    await index_manager.create_index_if_not_exists()
    await eval_repo.create_tables()

    yield

    await os_client.close()
    await eval_repo.close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Secure Hybrid RAG Search",
        description="Fintech knowledge base with lexical, vector, and hybrid retrieval.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # TODO: restrict to known origins in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()
