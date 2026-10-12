"""Rebuild the FAISS knowledge base from the saved PDFs and the recorded URLs.

Use this after changing EMBEDDING_MODEL (the embedding-model guardrail locks the old
index), or to rebuild a damaged index. Run from the project root, with the backend stopped:

    python -m app.reindex          # asks for confirmation
    python -m app.reindex --yes    # no prompt

The old index is kept as a dated backup folder next to it. With the same embedding
model, the embedding cache makes a rebuild free (no API calls).
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.embedding_cache import CachedEmbeddings
from app.ingestion import IngestionService
from app.main import build_embeddings
from app.schemas import IngestItemResult
from app.vector_store import VectorStoreManager, read_manifest

logger = logging.getLogger(__name__)


@dataclass
class RebuildReport:
    backup_dir: Path | None = None
    results: list[IngestItemResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    restored: bool = False
    chunks: int = 0

    @property
    def ingested(self) -> int:
        return sum(1 for r in self.results if r.status == "ingested")


def _find_upload(settings: Settings, content_hash: str) -> Path | None:
    matches = sorted(settings.upload_dir.glob(f"{content_hash[:12]}_*"))
    return matches[0] if matches else None


def rebuild_index(settings: Settings, embeddings: Embeddings) -> RebuildReport:
    """Back up the current index, then re-ingest every recorded source into a new one."""
    report = RebuildReport()
    sources = sorted(
        read_manifest(settings.faiss_index_dir).get("sources", {}).values(),
        key=lambda s: s.get("ingested_at", ""),
    )
    if not sources:
        return report

    index_dir = settings.faiss_index_dir
    if index_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report.backup_dir = index_dir.with_name(f"{index_dir.name}.backup-{stamp}")
        index_dir.rename(report.backup_dir)

    store = VectorStoreManager(embeddings, index_dir, settings.embedding_signature)
    service = IngestionService(settings, store)

    for source in sources:
        name = source["display_name"]
        if source["source_type"] == "pdf":
            path = _find_upload(settings, source.get("content_hash", ""))
            if path is None:
                report.missing.append(f"{name} (saved copy not found in {settings.upload_dir})")
                continue
            report.results.append(service.ingest_pdf(name, path.read_bytes()))
        else:
            report.results.append(service.ingest_url(name))

    report.chunks = store.num_chunks
    if report.chunks == 0 and report.backup_dir is not None:
        # Nothing could be rebuilt: put the old index back rather than leave nothing.
        shutil.rmtree(index_dir, ignore_errors=True)
        report.backup_dir.rename(index_dir)
        report.backup_dir = None
        report.restored = True
    return report


def main() -> int:
    settings = get_settings()
    logging.basicConfig(level="WARNING", format="%(levelname)s | %(message)s")
    manifest = read_manifest(settings.faiss_index_dir)
    sources = manifest.get("sources", {})
    if not sources:
        print(f"No knowledge base found at {settings.faiss_index_dir}; nothing to rebuild.")
        return 0

    print(f"Knowledge base : {settings.faiss_index_dir}  ({len(sources)} sources)")
    print(f"Built with     : {manifest.get('embedding_signature', 'unknown')}")
    print(f"Rebuild with   : {settings.embedding_signature}")
    print("URLs will be downloaded again; PDFs are read from the saved uploads.")
    if "--yes" not in sys.argv:
        if input("Stop the backend first. Rebuild now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1

    embeddings = build_embeddings(settings)
    report = rebuild_index(settings, embeddings)

    print()
    for r in report.results:
        mark = {"ingested": "OK  ", "skipped_duplicate": "SKIP", "failed": "FAIL"}[r.status]
        print(f"  {mark} {r.source} - {r.message}")
    for m in report.missing:
        print(f"  MISS {m}")
    print()
    if isinstance(embeddings, CachedEmbeddings):
        print(f"Embeddings reused from cache: {embeddings.hits}, newly computed: {embeddings.misses}")
    if report.restored:
        print("Nothing could be rebuilt, so the previous knowledge base was restored unchanged.")
        return 1
    print(f"Rebuilt {report.ingested} of {len(sources)} sources ({report.chunks} chunks).")
    if report.backup_dir:
        print(f"Old knowledge base kept at {report.backup_dir} - delete it once you are happy.")
    print("Start the backend again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
