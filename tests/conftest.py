"""Shared fixtures. All model calls are replaced with offline fakes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable

from app.config import Settings
from app.main import create_app


def make_pdf(pages: list[str]) -> bytes:
    """Build a small, valid, text-based PDF (one string per page) without extra deps."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        lines = [esc(line) for line in text.split("\n")] if text else []
        ops = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
        stream = ops.encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


WHEAT_PAGE = (
    "Wheat is sown in India during the rabi season, from late October to November.\n"
    "Recommended seed rate for timely sown irrigated wheat is 100 kg per hectare.\n"
    "Apply 120 kg nitrogen, 60 kg phosphorus and 40 kg potash per hectare."
)
RICE_PAGE = (
    "Paddy transplanting is done when seedlings are 21 to 25 days old.\n"
    "Maintain 2 to 5 cm standing water during the vegetative stage of rice."
)


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf([WHEAT_PAGE, "", RICE_PAGE])  # page 2 is intentionally blank


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        embedding_provider="fake",
        fake_embedding_dim=32,
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        faiss_index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
        embedding_cache_dir=tmp_path / "cache",
        guardrail_topic_check=False,  # enabled explicitly in the guardrail tests
        log_level="WARNING",
    )


@pytest.fixture
def make_client(settings: Settings) -> Callable[..., TestClient]:
    """Factory: ``make_client(responses=[...])`` or ``make_client(llm=runnable)``."""

    def _make(
        responses: list[str] | None = None,
        llm: Runnable | None = None,
        guard_llm: Runnable | None = None,
        embeddings=None,
    ) -> TestClient:
        model = llm or FakeListChatModel(responses=responses or ["INSUFFICIENT_CONTEXT"])
        app = create_app(
            settings,
            embeddings=embeddings or DeterministicFakeEmbedding(size=32),
            llm=model,
            guard_llm=guard_llm,
        )
        return TestClient(app)

    return _make


class CountingEmbedding(DeterministicFakeEmbedding):
    """Fake embedding model that counts how many texts it was asked to embed."""

    calls: int = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.calls += 1
        return super().embed_query(text)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
