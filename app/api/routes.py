"""HTTP endpoints: health, ingestion, querying and source listing."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.embedding_cache import CachedEmbeddings
from app.errors import describe_model_error
from app.ingestion import IngestionService
from app.retrieval import QAService
from app.vector_store import VectorStoreError
from app.schemas import (
    HealthResponse,
    IngestItemResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    SourcesResponse,
    UrlIngestRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILES_PER_REQUEST = 20


# --- Dependencies (overridable in tests) ----------------------------------
def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa


def _summarise(results: list[IngestItemResult], service: IngestionService) -> IngestResponse:
    return IngestResponse(
        results=results,
        total_chunks_added=sum(r.chunks_added for r in results),
        index_chunks=service.store.num_chunks,
    )


# --- Endpoints -------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, service: IngestionService = Depends(get_ingestion_service)) -> HealthResponse:
    """Liveness check that also reports index size and configured providers."""
    settings = request.app.state.settings
    embeddings = request.app.state.embeddings
    guardrails = [
        name for name, on in (
            ("prompt_injection", settings.guardrail_block_injection),
            ("topic_check", settings.guardrail_topic_check),
            ("require_citations", settings.guardrail_require_citations),
            ("embedding_model_lock", True),
        ) if on
    ]
    return HealthResponse(
        index_status=service.store.status,
        index_message=service.store.status_message,
        index_chunks=service.store.num_chunks,
        sources=len(service.store.list_sources()),
        embedding_cache_vectors=embeddings.count() if isinstance(embeddings, CachedEmbeddings) else 0,
        llm_provider=f"{settings.llm_provider}:{settings.chat_model}",
        embedding_provider=f"{settings.embedding_provider}:{settings.embedding_model}",
        guardrails=guardrails,
    )


@router.get("/sources", response_model=SourcesResponse, tags=["ingestion"])
def list_sources(service: IngestionService = Depends(get_ingestion_service)) -> SourcesResponse:
    """List every source currently in the knowledge base."""
    return SourcesResponse(sources=[SourceInfo(**s) for s in service.store.list_sources()])


@router.post("/ingest/pdf", response_model=IngestResponse, tags=["ingestion"])
async def ingest_pdf(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest uploaded PDFs. Each file gets its own status in ``results``:
    a bad file is reported as ``failed`` without blocking the others."""
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Upload at most {MAX_FILES_PER_REQUEST} files per request."
        )

    limit = service.settings.max_upload_bytes
    results: list[IngestItemResult] = []
    for upload in files:
        filename = upload.filename or "upload.pdf"
        data = await upload.read(limit + 1)  # read one extra byte to detect oversize files
        await upload.close()
        if len(data) > limit:
            results.append(
                IngestItemResult(
                    source=filename, source_type="pdf", status="failed",
                    message=f"File exceeds the {service.settings.max_upload_mb} MB upload limit.",
                )
            )
            continue
        # Parsing and embedding are blocking; keep the event loop free.
        results.append(await run_in_threadpool(service.ingest_pdf, filename, data))
    return _summarise(results, service)


@router.post("/ingest/url", response_model=IngestResponse, tags=["ingestion"])
def ingest_url(
    payload: UrlIngestRequest, service: IngestionService = Depends(get_ingestion_service)
) -> IngestResponse:
    """Fetch and ingest web pages (HTML, plain text or directly linked PDFs)."""
    results: list[IngestItemResult] = []
    seen: set[str] = set()
    for url in payload.urls:
        key = str(url)
        if key in seen:
            continue
        seen.add(key)
        results.append(service.ingest_url(key))
    return _summarise(results, service)


@router.post("/query", response_model=QueryResponse, tags=["qa"])
def query(payload: QueryRequest, qa: QAService = Depends(get_qa_service)) -> QueryResponse:
    """Answer a question from the indexed sources, with citations and guardrails."""
    if qa.store.status_message:  # embedding-model guardrail
        raise HTTPException(status.HTTP_409_CONFLICT, qa.store.status_message)
    if qa.store.is_empty:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The knowledge base is empty. Ingest PDFs or URLs before asking questions.",
        )
    try:
        return qa.answer(payload.question, payload.top_k)
    except VectorStoreError as exc:  # e.g. vector size changed under the same model name
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"The model request failed ({exc.__class__.__name__}) - {describe_model_error(exc)}",
        ) from exc
