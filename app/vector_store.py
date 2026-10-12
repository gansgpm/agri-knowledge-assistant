"""Persistent FAISS vector store plus a manifest of ingested sources.

Layout of ``FAISS_INDEX_DIR``::

    index.faiss     # FAISS vectors (written by LangChain)
    index.pkl       # docstore + id mapping (written by LangChain)
    manifest.json   # embedding signature + registry of ingested sources

The manifest records which embedding model built the index (so an index is
never queried with incompatible vectors) and each source's content hash
(so duplicates are detected even when a PDF is re-uploaded under a new name).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
INDEX_NAME = "index"
MANIFEST_VERSION = 1


class VectorStoreError(RuntimeError):
    """The index on disk is unusable (e.g. built with a different embedding model)."""


class IndexLockedError(VectorStoreError):
    """Guardrail: the index was built with a different embedding model, so it is locked
    against both reads and writes until it is rebuilt (``python -m app.reindex``)."""


class EmbeddingDimensionError(VectorStoreError):
    """Guardrail: the embedding model returned vectors of a different size than the index."""


def read_manifest(index_dir: Path) -> dict[str, Any]:
    """Read an index manifest without loading the index (empty dict if absent)."""
    path = Path(index_dir) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class DuplicateSourceError(Exception):
    """The source (same URL or same content) is already in the index."""

    def __init__(self, existing: dict[str, Any]):
        self.existing = existing
        super().__init__(f"Already ingested as '{existing.get('display_name')}'")


class VectorStoreManager:
    """Thread-safe wrapper around a LangChain FAISS store persisted on disk."""

    def __init__(self, embeddings: Embeddings, index_dir: Path, embedding_signature: str):
        self._embeddings = embeddings
        self._index_dir = Path(index_dir)
        self._signature = embedding_signature
        self._lock = threading.RLock()
        self._store: FAISS | None = None
        self._sources: dict[str, dict[str, Any]] = {}
        self._locked_message: str | None = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        manifest_path = self._index_dir / MANIFEST_FILE
        index_path = self._index_dir / f"{INDEX_NAME}.faiss"
        if not index_path.exists():
            logger.info("No FAISS index at %s yet; it will be created on first ingest.", self._index_dir)
            return
        manifest = read_manifest(self._index_dir)
        stored_signature = manifest.get("embedding_signature")
        if stored_signature and stored_signature != self._signature:
            # Guardrail: never mix vectors from two embedding models. The app still starts
            # (so the UI can explain the problem), but the index is locked.
            self._sources = manifest.get("sources", {})
            self._locked_message = (
                f"The knowledge base was built with embedding model '{stored_signature}', but the "
                f"current setting is '{self._signature}'. Vectors from different models cannot be "
                "compared, so the knowledge base is locked. Either restore the original "
                "EMBEDDING_MODEL in .env, or rebuild with:  python -m app.reindex"
            )
            logger.error("Index locked: %s", self._locked_message)
            return
        # The pickle is produced by this application only, so deserialising it is safe.
        self._store = FAISS.load_local(
            str(self._index_dir), self._embeddings, INDEX_NAME, allow_dangerous_deserialization=True
        )
        self._sources = manifest.get("sources", {})
        logger.info(
            "Loaded FAISS index: %d chunks from %d sources", self._store.index.ntotal, len(self._sources)
        )

    def _save(self) -> None:
        """Write index + manifest to a temp dir, then swap it in atomically-ish."""
        assert self._store is not None
        self._index_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = self._index_dir.with_name(self._index_dir.name + ".tmp")
        backup_dir = self._index_dir.with_name(self._index_dir.name + ".bak")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        self._store.save_local(str(tmp_dir), INDEX_NAME)
        manifest = {
            "version": MANIFEST_VERSION,
            "embedding_signature": self._signature,
            "embedding_dim": int(self._store.index.d),
            "sources": self._sources,
        }
        (tmp_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.rmtree(backup_dir, ignore_errors=True)
        if self._index_dir.exists():
            self._index_dir.rename(backup_dir)
        tmp_dir.rename(self._index_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Sources registry
    # ------------------------------------------------------------------
    def find_duplicate(self, source_id: str, content_hash: str) -> dict[str, Any] | None:
        """Return the existing record if the source id or content hash is known."""
        with self._lock:
            if source_id in self._sources:
                return self._sources[source_id]
            for record in self._sources.values():
                if record.get("content_hash") == content_hash:
                    return record
            return None

    def list_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._sources.values(), key=lambda r: r.get("ingested_at", ""), reverse=True)

    @property
    def status(self) -> str:
        if self._locked_message:
            return "model_mismatch"
        return "empty" if self.is_empty else "ok"

    @property
    def status_message(self) -> str | None:
        return self._locked_message

    def _guard_unlocked(self) -> None:
        if self._locked_message:
            raise IndexLockedError(self._locked_message)

    def _guard_dimension(self, dim: int) -> None:
        if self._store is not None and dim != self._store.index.d:
            raise EmbeddingDimensionError(
                f"The embedding model returned {dim}-dimensional vectors, but the knowledge base "
                f"stores {self._store.index.d}-dimensional ones. The embedding model has changed; "
                "restore it in .env or rebuild with:  python -m app.reindex"
            )

    @property
    def num_chunks(self) -> int:
        with self._lock:
            return int(self._store.index.ntotal) if self._store is not None else 0

    @property
    def is_empty(self) -> bool:
        return self.num_chunks == 0

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------
    def add_source(
        self,
        chunks: list[Document],
        *,
        source_id: str,
        source_type: str,
        display_name: str,
        content_hash: str,
        pages: int | None,
    ) -> int:
        """Embed chunks and add them to the index under one source record.

        Embedding happens outside the lock (it is the slow, network-bound part);
        the index mutation and save happen inside it.
        """
        if not chunks:
            return 0
        self._guard_unlocked()
        texts = [c.page_content for c in chunks]
        vectors = self._embeddings.embed_documents(texts)
        if len({len(v) for v in vectors}) != 1:
            raise EmbeddingDimensionError("The embedding model returned vectors of mixed sizes.")
        ids = [c.metadata["chunk_id"] for c in chunks]
        metadatas = [c.metadata for c in chunks]

        with self._lock:
            existing = self.find_duplicate(source_id, content_hash)
            if existing is not None:  # lost a race with a concurrent request
                raise DuplicateSourceError(existing)
            self._guard_dimension(len(vectors[0]))
            pairs = list(zip(texts, vectors))
            if self._store is None:
                self._store = FAISS.from_embeddings(pairs, self._embeddings, metadatas=metadatas, ids=ids)
            else:
                self._store.add_embeddings(pairs, metadatas=metadatas, ids=ids)
            self._sources[source_id] = {
                "source_id": source_id,
                "source_type": source_type,
                "display_name": display_name,
                "content_hash": content_hash,
                "chunks": len(chunks),
                "pages": pages,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._save()
        logger.info("Indexed %d chunks for %s", len(chunks), display_name)
        return len(chunks)

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return up to ``k`` (document, L2 distance) pairs, closest first."""
        self._guard_unlocked()
        if self.is_empty:
            return []
        vector = self._embeddings.embed_query(query)
        with self._lock:
            assert self._store is not None
            self._guard_dimension(len(vector))
            return self._store.similarity_search_with_score_by_vector(vector, k=k)
