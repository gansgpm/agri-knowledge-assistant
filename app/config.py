"""Application configuration loaded from environment variables and `.env`.

Every tunable value lives here so the rest of the code never reads
``os.environ`` directly and never contains secrets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "fake"]


class Settings(BaseSettings):
    """Typed settings for the backend (and shared defaults for the UI)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Model providers -------------------------------------------------
    llm_provider: ProviderName = "openai"
    embedding_provider: ProviderName = "openai"
    openai_api_key: SecretStr | None = None
    # Set this to use any OpenAI-compatible server (Ollama, vLLM, Groq, LM Studio...).
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(default=60, ge=1)
    fake_embedding_dim: int = Field(default=256, ge=8)

    # --- Chunking --------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Retrieval -------------------------------------------------------
    top_k: int = Field(default=4, ge=1, le=20)
    # Optional L2-distance cut-off; chunks farther than this are ignored.
    # With normalised embeddings (OpenAI), distances range roughly 0-2.
    max_distance: float | None = Field(default=None, gt=0)

    # --- Storage ---------------------------------------------------------
    faiss_index_dir: Path = Path("data/faiss_index")
    upload_dir: Path = Path("data/uploads")
    save_uploads: bool = True

    # --- Ingestion limits ------------------------------------------------
    max_upload_mb: int = Field(default=25, ge=1)
    max_url_mb: int = Field(default=15, ge=1)
    url_timeout_seconds: int = Field(default=20, ge=1)
    min_chars_per_source: int = Field(default=50, ge=1)
    # Block localhost / private-network URLs (protects against SSRF).
    allow_private_urls: bool = False

    # --- Guardrails ------------------------------------------------------
    # Block questions that try to override the assistant's instructions.
    guardrail_block_injection: bool = True
    # Ask the chat model whether a question is about agriculture before answering.
    guardrail_topic_check: bool = True
    # Withhold answers that cite none of the retrieved passages.
    guardrail_require_citations: bool = True

    # --- Embedding cache -------------------------------------------------
    # Every embedding is stored on disk, keyed by model + text, and reused.
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = Path("data/embedding_cache")

    # --- Misc ------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _lowercase_provider(cls, value: str) -> str:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("max_distance", mode="before")
    @classmethod
    def _blank_distance_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_url_bytes(self) -> int:
        return self.max_url_mb * 1024 * 1024

    @property
    def embedding_signature(self) -> str:
        """Identifies the embedding space; an index built with one signature
        cannot be queried with another."""
        if self.embedding_provider == "fake":
            return f"fake:{self.fake_embedding_dim}"
        return f"{self.embedding_provider}:{self.openai_base_url or 'default'}:{self.embedding_model}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()
