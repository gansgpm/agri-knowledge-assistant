"""Factories for the embedding model and the chat model.

This is the only module that knows about concrete providers. To add a new
provider (e.g. Google, Anthropic, Bedrock, HuggingFace), add a branch to
:func:`get_embeddings` and/or :func:`get_chat_model` that returns a LangChain
``Embeddings`` / ``BaseChatModel`` and extend ``ProviderName`` in ``config.py``.
Everything else in the app depends only on the LangChain interfaces.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings

logger = logging.getLogger(__name__)


class ProviderConfigError(RuntimeError):
    """Raised when a provider is selected but not configured correctly."""


def _require_openai_key(settings: Settings) -> str:
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise ProviderConfigError(
            "OPENAI_API_KEY is not set. Add it to your .env file "
            "(for local OpenAI-compatible servers such as Ollama any non-empty value works)."
        )
    return settings.openai_api_key.get_secret_value()


def get_embeddings(settings: Settings) -> Embeddings:
    """Build the embedding model selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.embedding_provider
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI-compatible embeddings: %s", settings.embedding_model)
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=_require_openai_key(settings),
            base_url=settings.openai_base_url,
            # Chunks (CHUNK_SIZE <= 8000 chars) are far below the 8191-token limit, so
            # token-length checking is unnecessary. Disabling it also avoids downloading
            # the tiktoken file (often blocked on office networks) and keeps Ollama working.
            check_embedding_ctx_length=False,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
    if provider == "fake":
        logger.warning("Using FAKE embeddings - retrieval quality is meaningless (tests only).")
        return DeterministicFakeEmbedding(size=settings.fake_embedding_dim)
    raise ProviderConfigError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model selected by ``LLM_PROVIDER``."""
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI-compatible chat model: %s", settings.chat_model)
        return ChatOpenAI(
            model=settings.chat_model,
            api_key=_require_openai_key(settings),
            base_url=settings.openai_base_url,
            temperature=settings.temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
    if provider == "fake":
        logger.warning("Using FAKE chat model - answers are canned (tests only).")
        return FakeListChatModel(responses=["INSUFFICIENT_CONTEXT"])
    raise ProviderConfigError(f"Unsupported LLM_PROVIDER: {provider!r}")
