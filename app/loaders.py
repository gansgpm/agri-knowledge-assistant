"""Turn raw inputs (PDF bytes, web URLs) into LangChain ``Document`` objects.

Each returned document is one PDF page or one web page, carrying metadata:
``source_type`` ("pdf" | "url"), ``source`` (filename or URL), ``title`` and,
for PDFs, a 1-based ``page`` number.
"""

from __future__ import annotations

import io
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings

logger = logging.getLogger(__name__)

USER_AGENT = "AgriKnowledgeAssistant/1.0 (+https://localhost; RAG ingestion bot)"
MAX_REDIRECTS = 5
_HTML_NOISE_TAGS = ("script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg", "iframe")


class DocumentLoadError(Exception):
    """A source could not be turned into text. The message is user-facing."""


@dataclass(frozen=True)
class DownloadedContent:
    """Raw response from a URL after redirects."""

    final_url: str
    content_type: str
    body: bytes
    encoding: str | None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_url(url: str) -> str:
    """Canonical form used for duplicate detection (lower-case host, no fragment)."""
    parts = urlsplit(url.strip())
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def validate_public_url(url: str, allow_private: bool) -> None:
    """Reject non-http(s) URLs and, unless allowed, hosts on private networks."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise DocumentLoadError("Only http:// and https:// URLs are supported.")
    host = parts.hostname
    if not host:
        raise DocumentLoadError("The URL has no host name.")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise DocumentLoadError(f"Could not resolve host '{host}'.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise DocumentLoadError(
                f"Host '{host}' resolves to a private or local address; "
                "set ALLOW_PRIVATE_URLS=true to allow this."
            )


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def load_pdf_bytes(data: bytes, source: str, source_type: str = "pdf") -> list[Document]:
    """Extract text page by page from PDF bytes.

    Empty pages (e.g. scanned images without OCR) are skipped. Raises
    :class:`DocumentLoadError` if the PDF is invalid, encrypted or has no text.
    """
    if not data:
        raise DocumentLoadError("The file is empty.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise DocumentLoadError("The file is not a valid PDF (missing %PDF header).")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty user password.
            if not reader.decrypt(""):
                raise DocumentLoadError("The PDF is password-protected.")
        pages = list(reader.pages)
    except DocumentLoadError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentLoadError(f"The PDF could not be read: {exc}") from exc

    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
    except Exception:  # metadata parsing is best-effort
        title = None

    documents: list[Document] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as exc:  # one broken page must not sink the whole file
            logger.warning("Skipping unreadable page %s of %s: %s", page_number, source, exc)
            continue
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_type": source_type,
                    "source": source,
                    "title": title or source,
                    "page": page_number,
                    "total_pages": len(pages),
                },
            )
        )

    if not documents:
        raise DocumentLoadError(
            "No extractable text found in the PDF (it may be a scanned image without OCR)."
        )
    logger.info("Extracted %d/%d text pages from %s", len(documents), len(pages), source)
    return documents


# --------------------------------------------------------------------------
# URL
# --------------------------------------------------------------------------
def _download(url: str, settings: Settings) -> DownloadedContent:
    """GET a URL, following redirects manually so every hop is validated."""
    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.5"})
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_url(current, settings.allow_private_urls)
        try:
            response = session.get(
                current, timeout=settings.url_timeout_seconds, allow_redirects=False, stream=True
            )
        except requests.Timeout as exc:
            raise DocumentLoadError("The request timed out.") from exc
        except requests.RequestException as exc:
            raise DocumentLoadError(f"The URL could not be reached: {exc.__class__.__name__}.") from exc

        with response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise DocumentLoadError("Redirect without a Location header.")
                current = urljoin(current, location)
                continue
            if response.status_code >= 400:
                raise DocumentLoadError(f"The server returned HTTP {response.status_code}.")

            body = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > settings.max_url_bytes:
                    raise DocumentLoadError(
                        f"The page is larger than the {settings.max_url_mb} MB limit."
                    )
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            charset = response.encoding if "charset" in response.headers.get("Content-Type", "").lower() else None
            return DownloadedContent(current, content_type, bytes(body), charset)
    raise DocumentLoadError("Too many redirects.")


def html_to_text(html: bytes | str, encoding: str | None = None) -> tuple[str, str | None]:
    """Extract readable text and the <title> from HTML."""
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None)
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_HTML_NOISE_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return clean_text(root.get_text(separator="\n")), title or None


def load_url(url: str, settings: Settings) -> list[Document]:
    """Download a URL and return its readable content as documents.

    HTML pages become one document; PDFs linked directly are split per page.
    """
    downloaded = _download(url, settings)
    body, ctype = downloaded.body, downloaded.content_type

    if ctype == "application/pdf" or body.lstrip()[:5].startswith(b"%PDF"):
        return load_pdf_bytes(body, source=url, source_type="url")

    if ctype in ("text/html", "application/xhtml+xml", "") or "html" in ctype:
        text, title = html_to_text(body, downloaded.encoding)
    elif ctype.startswith("text/"):
        text, title = clean_text(body.decode(downloaded.encoding or "utf-8", errors="replace")), None
    else:
        raise DocumentLoadError(f"Unsupported content type '{ctype}'. Use an HTML page, text or PDF.")

    if len(text) < settings.min_chars_per_source:
        raise DocumentLoadError(
            "The page has no readable text (it may be empty or rendered by JavaScript)."
        )
    logger.info("Fetched %d characters from %s", len(text), url)
    return [
        Document(
            page_content=text,
            metadata={"source_type": "url", "source": url, "title": title or url},
        )
    ]
