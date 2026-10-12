"""Ingestion pipeline: validate -> de-duplicate -> load -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

from app.chunking import split_documents
from app.config import Settings
from app.errors import describe_model_error
from app.loaders import DocumentLoadError, load_pdf_bytes, load_url, normalize_url
from app.schemas import IngestItemResult
from app.vector_store import DuplicateSourceError, VectorStoreError, VectorStoreManager

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters from an uploaded file name."""
    base = Path(name or "upload.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return cleaned or "upload.pdf"


class IngestionService:
    """Coordinates loaders, chunking and the vector store."""

    def __init__(self, settings: Settings, store: VectorStoreManager):
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------
    def ingest_pdf(self, filename: str, data: bytes) -> IngestItemResult:
        """Ingest one uploaded PDF. Never raises for bad input; returns a result."""
        name = safe_filename(filename)

        def failed(message: str) -> IngestItemResult:
            logger.warning("PDF '%s' rejected: %s", name, message)
            return IngestItemResult(source=name, source_type="pdf", status="failed", message=message)

        if not name.lower().endswith(".pdf"):
            return failed("Only .pdf files are accepted.")
        if len(data) > self.settings.max_upload_bytes:
            return failed(f"File exceeds the {self.settings.max_upload_mb} MB upload limit.")

        content_hash = hashlib.sha256(data).hexdigest()
        source_id = f"pdf:{content_hash[:16]}"
        if (dup := self.store.find_duplicate(source_id, content_hash)) is not None:
            return self._duplicate(name, "pdf", dup)

        try:
            pages = load_pdf_bytes(data, source=name)
        except DocumentLoadError as exc:
            return failed(str(exc))

        result = self._index(pages, source_id, "pdf", name, content_hash, page_count=len(pages))
        if result.status == "ingested" and self.settings.save_uploads:
            self._save_upload(name, content_hash, data)
        return result

    def ingest_url(self, url: str) -> IngestItemResult:
        """Fetch and ingest one URL. Never raises for bad input; returns a result."""
        normalized = normalize_url(url)
        source_id = f"url:{normalized}"

        if (dup := self.store.find_duplicate(source_id, content_hash="")) is not None:
            return self._duplicate(normalized, "url", dup)

        try:
            documents = load_url(normalized, self.settings)
        except DocumentLoadError as exc:
            logger.warning("URL '%s' rejected: %s", normalized, exc)
            return IngestItemResult(source=normalized, source_type="url", status="failed", message=str(exc))

        content_hash = hashlib.sha256(
            "\n".join(d.page_content for d in documents).encode("utf-8")
        ).hexdigest()
        if (dup := self.store.find_duplicate(source_id, content_hash)) is not None:
            return self._duplicate(normalized, "url", dup)

        pdf_pages = len(documents) if documents and "page" in documents[0].metadata else None
        return self._index(documents, source_id, "url", normalized, content_hash, page_count=pdf_pages)

    # ------------------------------------------------------------------
    def _index(
        self,
        documents: list[Document],
        source_id: str,
        source_type: str,
        display_name: str,
        content_hash: str,
        page_count: int | None,
    ) -> IngestItemResult:
        chunks = split_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap, source_id
        )
        if not chunks:
            return IngestItemResult(
                source=display_name, source_type=source_type, status="failed",
                message="No text chunks could be created from this source.",
            )
        try:
            added = self.store.add_source(
                chunks,
                source_id=source_id,
                source_type=source_type,
                display_name=display_name,
                content_hash=content_hash,
                pages=page_count,
            )
        except DuplicateSourceError as exc:
            return self._duplicate(display_name, source_type, exc.existing)
        except VectorStoreError as exc:  # embedding-model guardrail
            logger.error("Ingestion of %s blocked: %s", display_name, exc)
            return IngestItemResult(
                source=display_name, source_type=source_type, status="failed", message=str(exc),
            )
        except Exception as exc:  # embedding API errors, disk errors, ...
            logger.exception("Failed to index %s", display_name)
            return IngestItemResult(
                source=display_name, source_type=source_type, status="failed",
                message=f"Embedding failed - {describe_model_error(exc)}",
            )
        return IngestItemResult(
            source=display_name,
            source_type=source_type,
            status="ingested",
            chunks_added=added,
            pages=page_count,
            message=f"Indexed {added} chunks" + (f" from {page_count} pages." if page_count else "."),
        )

    @staticmethod
    def _duplicate(name: str, source_type: str, existing: dict) -> IngestItemResult:
        return IngestItemResult(
            source=name,
            source_type=source_type,  # type: ignore[arg-type]
            status="skipped_duplicate",
            message=f"Already in the knowledge base as '{existing.get('display_name')}'.",
        )

    def _save_upload(self, name: str, content_hash: str, data: bytes) -> None:
        try:
            self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.upload_dir / f"{content_hash[:12]}_{name}").write_bytes(data)
        except OSError:
            logger.exception("Could not save uploaded file %s (ingestion still succeeded)", name)
