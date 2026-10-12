"""Ingestion validation tests (no network, no real models)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import loaders
from app.config import Settings
from app.loaders import DocumentLoadError, DownloadedContent, html_to_text, load_pdf_bytes, normalize_url
from tests.conftest import make_pdf


def _upload(client: TestClient, name: str, data: bytes, ctype: str = "application/pdf"):
    return client.post("/ingest/pdf", files=[("files", (name, data, ctype))])


# --- Settings ---------------------------------------------------------------
def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)


# --- PDF loader -------------------------------------------------------------
def test_pdf_loader_extracts_pages_and_skips_blank(sample_pdf: bytes):
    docs = load_pdf_bytes(sample_pdf, source="crops.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert docs[0].metadata["source_type"] == "pdf"
    assert "100 kg per hectare" in docs[0].page_content


@pytest.mark.parametrize(
    "data, message",
    [(b"", "empty"), (b"hello, not a pdf", "not a valid PDF"), (b"%PDF-1.4\ngarbage", None)],
)
def test_pdf_loader_rejects_bad_files(data: bytes, message: str | None):
    with pytest.raises(DocumentLoadError, match=message):
        load_pdf_bytes(data, source="bad.pdf")


def test_pdf_loader_rejects_pdf_without_text():
    with pytest.raises(DocumentLoadError, match="No extractable text"):
        load_pdf_bytes(make_pdf(["", ""]), source="scan.pdf")


# --- /ingest/pdf ------------------------------------------------------------
def test_ingest_pdf_success_and_metadata(client: TestClient, sample_pdf: bytes, settings: Settings):
    res = _upload(client, "crops.pdf", sample_pdf)
    assert res.status_code == 200
    item = res.json()["results"][0]
    assert item["status"] == "ingested"
    assert item["chunks_added"] > 0 and item["pages"] == 2
    assert (settings.faiss_index_dir / "index.faiss").exists()

    sources = client.get("/sources").json()["sources"]
    assert sources[0]["display_name"] == "crops.pdf"
    assert client.get("/health").json()["index_chunks"] == item["chunks_added"]


def test_ingest_duplicate_pdf_is_skipped_even_if_renamed(client: TestClient, sample_pdf: bytes):
    assert _upload(client, "crops.pdf", sample_pdf).json()["results"][0]["status"] == "ingested"
    second = _upload(client, "crops-copy.pdf", sample_pdf).json()
    assert second["results"][0]["status"] == "skipped_duplicate"
    assert second["total_chunks_added"] == 0


def test_ingest_invalid_and_wrong_type_files_fail_individually(client: TestClient, sample_pdf: bytes):
    res = client.post(
        "/ingest/pdf",
        files=[
            ("files", ("notes.txt", b"plain text", "text/plain")),
            ("files", ("broken.pdf", b"not really a pdf", "application/pdf")),
            ("files", ("good.pdf", sample_pdf, "application/pdf")),
        ],
    )
    assert res.status_code == 200
    statuses = [r["status"] for r in res.json()["results"]]
    assert statuses == ["failed", "failed", "ingested"]


def test_ingest_pdf_rejects_oversized_file(make_client, settings: Settings, sample_pdf: bytes):
    settings.max_upload_mb = 1
    client = make_client()
    big = sample_pdf + b"0" * (1024 * 1024)
    item = _upload(client, "big.pdf", big).json()["results"][0]
    assert item["status"] == "failed" and "limit" in item["message"]


def test_ingest_pdf_requires_files(client: TestClient):
    assert client.post("/ingest/pdf").status_code == 422


def test_index_persists_across_restarts(make_client, sample_pdf: bytes):
    first = make_client()
    _upload(first, "crops.pdf", sample_pdf)
    chunks = first.get("/health").json()["index_chunks"]

    restarted = make_client()  # new app instance, same index directory
    health = restarted.get("/health").json()
    assert health["index_chunks"] == chunks and health["sources"] == 1


# --- URL loader / /ingest/url -----------------------------------------------
HTML = b"""<html><head><title>Soil health</title><script>var x=1;</script></head>
<body><nav>Home | About</nav><main><h1>Soil testing</h1>
<p>Soil testing every three years helps farmers apply the right amount of fertiliser.</p>
<p>Collect samples from 0-15 cm depth in a zig-zag pattern across the field.</p></main>
<footer>Copyright</footer></body></html>"""


def test_html_to_text_strips_boilerplate():
    text, title = html_to_text(HTML)
    assert title == "Soil health"
    assert "zig-zag" in text
    assert "var x" not in text and "Home | About" not in text and "Copyright" not in text


def test_normalize_url():
    assert normalize_url("HTTPS://Example.ORG/guide/#top") == "https://example.org/guide"


def test_ingest_url_success_and_duplicate(client: TestClient, monkeypatch):
    def fake_download(url, settings):
        return DownloadedContent(url, "text/html", HTML, "utf-8")

    monkeypatch.setattr(loaders, "_download", fake_download)
    res = client.post("/ingest/url", json={"urls": ["https://example.org/soil", "https://example.org/soil"]})
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 1 and results[0]["status"] == "ingested"

    again = client.post("/ingest/url", json={"urls": ["https://EXAMPLE.org/soil/"]}).json()
    assert again["results"][0]["status"] == "skipped_duplicate"


def test_ingest_url_empty_page_fails(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        loaders, "_download",
        lambda url, s: DownloadedContent(url, "text/html", b"<html><body><div id='app'></div></body></html>", None),
    )
    item = client.post("/ingest/url", json={"urls": ["https://example.org/spa"]}).json()["results"][0]
    assert item["status"] == "failed" and "no readable text" in item["message"]


def test_ingest_url_unreachable_fails_gracefully(client: TestClient, monkeypatch):
    def boom(url, settings):
        raise DocumentLoadError("The URL could not be reached: ConnectionError.")

    monkeypatch.setattr(loaders, "_download", boom)
    item = client.post("/ingest/url", json={"urls": ["https://down.example.org"]}).json()["results"][0]
    assert item["status"] == "failed" and "could not be reached" in item["message"]


def test_ingest_url_blocks_private_addresses(client: TestClient):
    item = client.post("/ingest/url", json={"urls": ["http://127.0.0.1:8000/health"]}).json()["results"][0]
    assert item["status"] == "failed" and "private or local" in item["message"]


@pytest.mark.parametrize("payload", [{"urls": []}, {"urls": ["not a url"]}, {"urls": ["ftp://x.org/a"]}, {}])
def test_ingest_url_validates_input(client: TestClient, payload: dict):
    assert client.post("/ingest/url", json=payload).status_code == 422


def test_embedding_failure_reports_the_reason(make_client, sample_pdf: bytes, monkeypatch):
    from app.vector_store import VectorStoreManager

    class AuthenticationError(Exception):
        pass

    def fail(*args, **kwargs):
        raise AuthenticationError("Error code: 401 - Incorrect API key provided")

    monkeypatch.setattr(VectorStoreManager, "add_source", fail)
    item = _upload(make_client(), "crops.pdf", sample_pdf).json()["results"][0]
    assert item["status"] == "failed"
    assert "API key was rejected" in item["message"]
