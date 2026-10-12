"""FastAPI application factory.

Run with::

    uvicorn app.main:create_app --factory --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from app.api.routes import router
from app.config import Settings, get_settings
from app.embedding_cache import CachedEmbeddings
from app.ingestion import IngestionService
from app.logging_config import setup_logging
from app.providers import get_chat_model, get_embeddings
from app.retrieval import QAService
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings, embeddings: Embeddings | None = None) -> Embeddings:
    """The configured embedding model, wrapped in the on-disk cache when enabled."""
    embeddings = embeddings or get_embeddings(settings)
    if settings.embedding_cache_enabled:
        embeddings = CachedEmbeddings(embeddings, settings.embedding_cache_dir, settings.embedding_signature)
    return embeddings


def create_app(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    llm: Runnable | None = None,
    guard_llm: Runnable | None = None,
) -> FastAPI:
    """Build the app. Tests pass fake ``embeddings``/``llm``/``guard_llm`` to avoid network calls."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    embeddings = build_embeddings(settings, embeddings)
    llm = llm or get_chat_model(settings)
    store = VectorStoreManager(embeddings, settings.faiss_index_dir, settings.embedding_signature)

    app = FastAPI(
        title="Agriculture Knowledge Assistant API",
        version="1.0.0",
        description="RAG over agriculture PDFs and web pages with grounded, cited answers.",
    )
    app.state.settings = settings
    app.state.ingestion = IngestionService(settings, store)
    app.state.embeddings = embeddings
    app.state.qa = QAService(settings, store, llm, guard_llm=guard_llm)
    app.include_router(router)

    logger.info(
        "API ready | llm=%s:%s | embeddings=%s:%s | index=%s (%d chunks, status=%s)",
        settings.llm_provider, settings.chat_model,
        settings.embedding_provider, settings.embedding_model,
        settings.faiss_index_dir, store.num_chunks, store.status,
    )
    return app
