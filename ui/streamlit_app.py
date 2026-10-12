"""Streamlit front end for the Agriculture Knowledge Assistant.

Run with::

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
INGEST_TIMEOUT = 300  # embedding a large PDF can take a while
QUERY_TIMEOUT = 120

st.set_page_config(page_title="Agriculture Knowledge Assistant", page_icon="🌾", layout="wide")


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def _error_text(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail)


def api_call(method: str, path: str, timeout: int = 30, **kwargs: Any) -> tuple[dict | None, str | None]:
    """Call the backend. Returns (json, None) on success or (None, error message)."""
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        return None, f"Cannot reach the backend at {API_BASE_URL}. Is FastAPI running?"
    except requests.Timeout:
        return None, "The backend took too long to respond."
    if response.ok:
        return response.json(), None
    return None, _error_text(response)


def show_ingest_result(item: dict) -> None:
    label = f"**{item['source']}** — {item['message']}"
    if item["status"] == "ingested":
        st.success(label, icon="✅")
    elif item["status"] == "skipped_duplicate":
        st.info(label, icon="♻️")
    else:
        st.error(label, icon="⚠️")


def format_citation(c: dict) -> str:
    where = c["source"]
    if c.get("page"):
        where += f" — page {c['page']}"
    return f"[{c['ref']}] {where}"


# --------------------------------------------------------------------------
# Sidebar: backend status + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🌾 Agri Assistant")
    health, health_err = api_call("GET", "/health", timeout=5)
    if health_err:
        st.error(health_err)
    else:
        if health.get("index_status") == "model_mismatch":
            st.error(f"**Knowledge base locked.** {health['index_message']}", icon="🔒")
        else:
            st.success("Backend online")
        c1, c2 = st.columns(2)
        c1.metric("Sources", health["sources"])
        c2.metric("Chunks", health["index_chunks"])
        st.caption(
            f"LLM: `{health['llm_provider']}`  \nEmbeddings: `{health['embedding_provider']}`  \n"
            f"Cached embeddings: {health.get('embedding_cache_vectors', 0)}"
        )
        if health.get("guardrails"):
            labels = {
                "prompt_injection": "Prompt-injection block",
                "topic_check": "Agriculture-only topic check",
                "require_citations": "Citations required",
                "embedding_model_lock": "Embedding-model lock",
            }
            st.caption("🛡️ Guardrails on: " + ", ".join(labels.get(g, g) for g in health["guardrails"]))
    st.divider()
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 4)
    if st.button("Clear chat history"):
        st.session_state.messages = []
    st.caption(f"API: {API_BASE_URL}")

st.title("Agriculture Knowledge Assistant")
st.caption("Answers come only from the PDFs and web pages you add, with citations.")

tab_ask, tab_add, tab_library = st.tabs(["💬 Ask", "📥 Add documents", "📚 Library"])

# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------
with tab_ask:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_answer(payload: dict) -> None:
        if payload["answerable"]:
            st.markdown(payload["answer"])
        elif payload.get("guardrail") in ("prompt_injection", "off_topic"):
            st.warning(payload["answer"], icon="🛡️")
        else:
            st.warning(payload["answer"], icon="🔎")
        for note in payload.get("notes", []):
            st.info(note, icon="⚠️")
        for warning in payload.get("warnings", []):
            st.caption(f"🛡️ {warning}")
        if payload["citations"]:
            st.markdown("**Sources**")
            for c in payload["citations"]:
                with st.expander(format_citation(c)):
                    if c["source_type"] == "url":
                        st.markdown(f"[Open page]({c['source']})")
                    if c.get("title") and c["title"] != c["source"]:
                        st.caption(c["title"])
                    st.write(c["snippet"])
                    st.caption(f"distance {c['distance']} (lower = closer)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif "error" in msg:
                st.error(msg["error"])
            else:
                render_answer(msg["payload"])

    question = st.chat_input("Ask about crops, soil, pests, irrigation, fertilisers…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your sources…"):
                data, err = api_call(
                    "POST", "/query", timeout=QUERY_TIMEOUT, json={"question": question, "top_k": top_k}
                )
            if err:
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "error": err})
            else:
                render_answer(data)
                st.session_state.messages.append({"role": "assistant", "payload": data})

# --------------------------------------------------------------------------
# Add documents
# --------------------------------------------------------------------------
with tab_add:
    left, right = st.columns(2)

    with left:
        st.subheader("Upload PDFs")
        uploads = st.file_uploader("Agriculture PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Ingest PDFs", type="primary", disabled=not uploads):
            progress = st.progress(0.0, text="Starting…")
            total = 0
            for i, f in enumerate(uploads, start=1):
                progress.progress((i - 1) / len(uploads), text=f"Processing {f.name} ({i}/{len(uploads)})…")
                data, err = api_call(
                    "POST", "/ingest/pdf", timeout=INGEST_TIMEOUT,
                    files=[("files", (f.name, f.getvalue(), "application/pdf"))],
                )
                if err:
                    st.error(f"**{f.name}** — {err}")
                    continue
                for item in data["results"]:
                    show_ingest_result(item)
                total += data["total_chunks_added"]
            progress.progress(1.0, text=f"Done — {total} new chunks indexed.")

    with right:
        st.subheader("Add web pages")
        raw_urls = st.text_area(
            "One URL per line",
            placeholder="https://www.fao.org/...\nhttps://agritech.tnau.ac.in/...",
            height=150,
        )
        urls = [u.strip() for u in raw_urls.splitlines() if u.strip()]
        if st.button("Ingest URLs", type="primary", disabled=not urls):
            progress = st.progress(0.0, text="Starting…")
            total = 0
            for i, url in enumerate(urls, start=1):
                progress.progress((i - 1) / len(urls), text=f"Fetching {url} ({i}/{len(urls)})…")
                if not url.lower().startswith(("http://", "https://")):
                    st.error(f"**{url}** — must start with http:// or https://")
                    continue
                data, err = api_call("POST", "/ingest/url", timeout=INGEST_TIMEOUT, json={"urls": [url]})
                if err:
                    st.error(f"**{url}** — {err}")
                    continue
                for item in data["results"]:
                    show_ingest_result(item)
                total += data["total_chunks_added"]
            progress.progress(1.0, text=f"Done — {total} new chunks indexed.")

# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
with tab_library:
    data, err = api_call("GET", "/sources")
    if err:
        st.error(err)
    elif not data["sources"]:
        st.info("No sources yet. Add PDFs or URLs in the **Add documents** tab.")
    else:
        st.dataframe(
            [
                {
                    "Source": s["display_name"],
                    "Type": s["source_type"].upper(),
                    "Pages": s["pages"] or "—",
                    "Chunks": s["chunks"],
                    "Added (UTC)": s["ingested_at"],
                }
                for s in data["sources"]
            ],
            width="stretch",
            hide_index=True,
        )
