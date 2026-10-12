"""Persistent embedding cache.

Every vector the embedding model returns is stored in a small SQLite file, keyed by
the embedding model (its "signature") and a SHA-256 hash of the text. Before calling
the model, the cache is checked; only texts never embedded before reach the API.

What this saves:
  * re-ingesting a source (e.g. after deleting the FAISS index, or `python -m app.reindex`
    with the same model) costs no embedding calls at all;
  * identical chunks shared by several documents are embedded once;
  * repeated questions do not re-embed the question.

Changing EMBEDDING_MODEL changes the signature, so old vectors are never reused for
a different model.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from array import array
from pathlib import Path

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DB_NAME = "embeddings.sqlite3"


class CachedEmbeddings(Embeddings):
    """Wraps any LangChain ``Embeddings`` with an on-disk cache."""

    def __init__(self, inner: Embeddings, cache_dir: Path, namespace: str):
        self.inner = inner
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cache_dir / DB_NAME, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, data BLOB NOT NULL)"
        )
        self._db.commit()

    # ------------------------------------------------------------------
    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{kind}\x00{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _encode(vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def _decode(blob: bytes) -> list[float]:
        values = array("f")
        values.frombytes(blob)
        return values.tolist()

    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            for start in range(0, len(keys), 500):  # SQLite parameter limit
                batch = keys[start:start + 500]
                marks = ",".join("?" * len(batch))
                for key, blob in self._db.execute(f"SELECT key, data FROM vectors WHERE key IN ({marks})", batch):
                    found[key] = self._decode(blob)
        return found

    def _put_many(self, items: dict[str, list[float]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (key, dim, data) VALUES (?, ?, ?)",
                [(k, len(v), self._encode(v)) for k, v in items.items()],
            )
            self._db.commit()

    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key("doc", t) for t in texts]
        cached = self._get_many(list(dict.fromkeys(keys)))
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in cached and key not in missing:
                missing[key] = text
        if missing:
            vectors = self.inner.embed_documents(list(missing.values()))
            fresh = dict(zip(missing.keys(), vectors))
            self._put_many(fresh)
            cached.update(fresh)
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        logger.info("Embedding cache: %d reused, %d newly embedded", len(texts) - len(missing), len(missing))
        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        cached = self._get_many([key])
        if key in cached:
            self.hits += 1
            return cached[key]
        vector = self.inner.embed_query(text)
        self._put_many({key: vector})
        self.misses += 1
        return vector

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
