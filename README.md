# 🌾 Agriculture Knowledge Assistant

Ask agriculture questions and get answers grounded **only** in the PDFs and web pages you add — with citations down to the page.

**Stack:** Streamlit (UI) → FastAPI (API) → LangChain (RAG) → FAISS (persistent local vector store) → any OpenAI-compatible embedding + chat model.

```
          ┌─────────────┐   HTTP    ┌──────────────────────────── FastAPI ─────────────────────────────┐
 User ──▶ │  Streamlit  │ ───────▶  │ /ingest/pdf, /ingest/url                                          │
          │  (ui/)      │           │   loaders.py → chunking.py → embeddings → vector_store.py (FAISS) │
          │             │ ◀───────  │ /query                                                            │
          └─────────────┘  answer + │   retrieval.py: similarity search → grounded prompt → chat model  │
                           citations└───────────────────────────────────────────────────────────────────┘
```

## Project structure

```
agri-rag/
├── app/
│   ├── api/routes.py      # FastAPI endpoints
│   ├── config.py          # Settings from .env (pydantic-settings)
│   ├── providers.py       # Embedding + chat model factories (swap providers here)
│   ├── loaders.py         # PDF text extraction, URL fetching + HTML cleaning
│   ├── chunking.py        # Overlapping chunking with metadata
│   ├── vector_store.py    # Persistent FAISS index + source manifest (duplicates)
│   ├── ingestion.py       # Ingestion pipeline orchestration
│   ├── retrieval.py       # Retrieval + grounded answering + citations
│   ├── guardrails.py      # Input + output guardrails
│   ├── embedding_cache.py # On-disk cache: each text is embedded once per model
│   ├── reindex.py         # `python -m app.reindex` - rebuild after a model change
│   ├── schemas.py         # Pydantic request/response models
│   ├── errors.py          # Readable explanations for model/API errors
│   ├── check.py           # `python -m app.check` - test your model settings
│   ├── logging_config.py
│   └── main.py            # App factory
├── ui/streamlit_app.py    # Streamlit front end
├── tests/                 # pytest suite (all model calls faked)
├── data/                  # FAISS index + uploads (git-ignored)
├── .env.example
├── requirements.txt
└── pytest.ini
```

## Setup

Requires Python 3.10+ (tested on 3.13).

```bash
cd agri-rag
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run

Terminal 1 — backend (from the project root):

```bash
uvicorn app.main:create_app --factory --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

Terminal 2 — frontend:

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501, add PDFs/URLs in **Add documents**, then ask in **Ask**.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | `openai` | `openai` (any OpenAI-compatible API) or `fake` (tests only) |
| `OPENAI_API_KEY` | — | API key (required for `openai`) |
| `OPENAI_BASE_URL` | blank | Point at an OpenAI-compatible server (Ollama, vLLM, Groq, LM Studio…) |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `TEMPERATURE` | `0.0` | Keep low for faithful answers |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters per chunk / overlap (overlap must be < size) |
| `TOP_K` | `4` | Chunks retrieved per question (UI slider overrides) |
| `MAX_DISTANCE` | blank | Optional L2-distance cut-off; drop weakly related chunks |
| `FAISS_INDEX_DIR` | `data/faiss_index` | Where the index is persisted |
| `UPLOAD_DIR`, `SAVE_UPLOADS` | `data/uploads`, `true` | Keep a copy of ingested PDFs |
| `MAX_UPLOAD_MB`, `MAX_URL_MB` | `25`, `15` | Size limits |
| `URL_TIMEOUT_SECONDS` | `20` | Fetch timeout per URL |
| `ALLOW_PRIVATE_URLS` | `false` | Block localhost/intranet URLs (SSRF protection) |
| `API_BASE_URL` | `http://localhost:8000` | Used by Streamlit to reach FastAPI |

### Provider notes

**OpenAI (default):** set `OPENAI_API_KEY`. Model names are just defaults — use any chat/embedding model your account has.

**Ollama (free, fully local):**
```bash
ollama pull llama3.1 && ollama pull nomic-embed-text
```
```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CHAT_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text
```
The same pattern works for any OpenAI-compatible endpoint (Groq, Together, vLLM, LM Studio, Azure OpenAI's v1 endpoint).

**Other LangChain providers** (Gemini, Anthropic, Bedrock, HuggingFace…): add a branch in `app/providers.py` that returns that provider's `Embeddings`/`BaseChatModel`, add its name to `ProviderName` in `app/config.py`, and `pip install` its LangChain package. Nothing else changes.

> ⚠️ **Changing the embedding model invalidates the index.** The backend refuses to start with a mismatched index and tells you so. Delete `data/faiss_index/` and re-ingest.

## How it behaves

- **Ingestion:** PDFs are read page by page with `pypdf`; blank pages are skipped and pages without text (scanned images) are reported. Web pages are fetched with redirects validated hop by hop, scripts/nav/footer stripped, and `<main>`/`<article>` preferred. Direct PDF links are handled as PDFs.
- **Chunks** carry `source_type`, `source` (filename or URL), `title`, `page` (when known), `chunk_index`, `start_index`.
- **Duplicates:** a PDF is identified by its SHA-256 content hash (re-uploading under a new name is still a duplicate); a URL by its normalised form and its content hash. Duplicates return `skipped_duplicate`.
- **Per-item results:** in a batch, one bad file or URL is reported as `failed` with a reason; the rest still ingest.
- **Persistence:** the index and `manifest.json` are written to a temp folder and swapped in, so a crash mid-save doesn't corrupt the existing index.
- **Grounding:** the prompt allows only the numbered context passages, requires `[n]` citations, and tells the model to reply `INSUFFICIENT_CONTEXT` when the sources don't cover the question — the API then returns `answerable: false` and a clear message, with no citations. Instructions found inside web pages are treated as data.

## Where the data is stored (nothing is embedded twice)

| Folder | What it holds |
|---|---|
| `data/faiss_index/` | The knowledge base: `index.faiss` (the vectors), `index.pkl` (chunk text + metadata), `manifest.json` (embedding model, vector size, list of sources) |
| `data/embedding_cache/` | Every embedding ever computed, keyed by embedding model + SHA-256 of the text |
| `data/uploads/` | A copy of each ingested PDF, used by `python -m app.reindex` |

Embeddings are computed once, when a source is first ingested, and saved. They are never recomputed when:

- the backend restarts (the index is loaded from disk);
- the same PDF is uploaded again, even under another name (duplicate check by content hash);
- the index is deleted or rebuilt with the same model (vectors come from the cache);
- the same text appears in several documents, or the same question is asked twice.

## Guardrails

| Stage | Guardrail | What happens |
|---|---|---|
| Question | Prompt-injection block | Questions like "ignore previous instructions" or "show your system prompt" are refused before any model call (`guardrail: "prompt_injection"`) |
| Question | Agriculture-only topic check | A short model call classifies the question; non-agriculture questions are refused (`guardrail: "off_topic"`). If the check itself fails, the question is allowed and grounding still applies |
| Answer | Grounding | The model may use only the retrieved passages and must reply `INSUFFICIENT_CONTEXT` when they don't answer |
| Answer | Citation validation | Citations that point to no retrieved passage (e.g. `[9]` when 4 were retrieved) are removed and reported in `warnings` |
| Answer | Citations required | An answer that cites nothing is withheld (`guardrail: "no_citation"`) |
| Answer | Coverage warning | If most statements carry no citation, a warning asks the user to verify them |
| Answer | Agrochemical safety note | Answers about pesticides, sprays or doses get a "follow the label, consult your extension officer" note in `notes` |
| Index | Embedding-model lock | If `EMBEDDING_MODEL` changes, the old index is locked (not used, not written to) and `/health` reports `index_status: "model_mismatch"`; the UI shows how to fix it |
| Index | Vector-size check | If the model returns vectors of a different size than the index (same name, different model), ingest and query are blocked with a clear message |

Each guardrail can be switched off in `.env` (`GUARDRAIL_*` settings) except the index locks.

### Changing the embedding model safely

1. Change `EMBEDDING_MODEL` in `.env` and stop the backend.
2. Run `python -m app.reindex`. It backs up the old index to `data/faiss_index.backup-<date>`, then rebuilds from the saved PDFs and by re-downloading the recorded URLs.
3. Start the backend. Delete the backup folder once you are happy.

With the **same** model, a rebuild is free: every vector comes from the embedding cache.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, index size, configured providers |
| `POST` | `/ingest/pdf` | Multipart upload, field `files` (repeatable) |
| `POST` | `/ingest/url` | JSON `{"urls": [...]}` (1–20 http/https URLs) |
| `POST` | `/query` | JSON `{"question": "...", "top_k": 4}` (`top_k` optional, 1–20) |
| `GET` | `/sources` | List ingested sources |

Errors: `422` invalid input, `400` too many files, `409` query before anything is ingested or while the index is locked by the embedding-model guardrail, `502` model/provider failure.

`/query` responses also carry `guardrail` (why an answer was blocked or withheld, else `null`), `warnings` and `notes`.

### curl examples

```bash
# Health
curl http://localhost:8000/health

# Ingest one or more PDFs
curl -X POST http://localhost:8000/ingest/pdf \
  -F "files=@docs/wheat_package_of_practices.pdf" \
  -F "files=@docs/rice_cultivation.pdf"

# Ingest URLs
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://en.wikipedia.org/wiki/Crop_rotation", "https://en.wikipedia.org/wiki/Drip_irrigation"]}'

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the benefits of crop rotation?", "top_k": 4}'

# List sources
curl http://localhost:8000/sources
```

Example `/ingest/pdf` response:

```json
{
  "results": [
    {"source": "wheat_package_of_practices.pdf", "source_type": "pdf", "status": "ingested",
     "chunks_added": 42, "pages": 12, "message": "Indexed 42 chunks from 12 pages."},
    {"source": "rice_cultivation.pdf", "source_type": "pdf", "status": "skipped_duplicate",
     "chunks_added": 0, "pages": null, "message": "Already in the knowledge base as 'rice_cultivation.pdf'."}
  ],
  "total_chunks_added": 42,
  "index_chunks": 118
}
```

Example `/query` response:

```json
{
  "question": "What is the seed rate for irrigated wheat?",
  "answer": "For timely sown irrigated wheat, the recommended seed rate is 100 kg per hectare [1].",
  "answerable": true,
  "citations": [
    {"ref": 1, "source": "wheat_package_of_practices.pdf", "source_type": "pdf", "title": "Wheat PoP",
     "page": 4, "snippet": "Recommended seed rate for timely sown irrigated wheat is 100 kg per hectare…",
     "distance": 0.6123}
  ]
}
```

When sources don't cover the question:

```json
{"question": "...", "answer": "The available sources do not contain enough information to answer this question. ...",
 "answerable": false, "citations": []}
```

## Tests

```bash
pytest
```

The suite (55 tests) runs offline: embeddings use LangChain's `DeterministicFakeEmbedding`, the chat model is `FakeListChatModel` or a `RunnableLambda`, and URL downloads are monkeypatched. It covers PDF validation (empty, non-PDF, corrupt, image-only, oversize, wrong extension), duplicates, persistence across restarts, HTML cleaning, unreachable/empty/private URLs, input validation, cited answers, the "not enough information" path, missing-citation handling and model failures (502), plus every guardrail (injection, topic, citation validation, safety note, model lock, vector-size check), the embedding cache (no re-embedding after the index is deleted, no re-embedding of repeated questions) and the rebuild command.

## Troubleshooting

**First step for any model error:** run the built-in check, which tests your `.env` settings directly and prints the exact cause:

```bash
python -m app.check
```


- **`OPENAI_API_KEY is not set`** at startup → create `.env` from `.env.example`.
- **"Knowledge base locked"** → you changed the embedding model; run `python -m app.reindex` (or restore the old `EMBEDDING_MODEL`).
- **"No extractable text found in the PDF"** → it's a scan; run OCR first (e.g. `ocrmypdf in.pdf out.pdf`).
- **"The page has no readable text"** → the site renders with JavaScript; save the page as PDF and upload it.
- **Answers often "not enough information"** → raise `TOP_K`, lower `CHUNK_SIZE`, or unset `MAX_DISTANCE`.
- **Run from the project root** so `app` is importable and `.env` / `data/` resolve correctly.
"# agri-knowledge-assistant" 
