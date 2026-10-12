"""Check that the configured embedding and chat models actually work.

Run from the project root:

    python -m app.check
"""

from __future__ import annotations

import sys
import traceback

from app.config import get_settings
from app.errors import describe_model_error
from app.providers import get_chat_model, get_embeddings
from app.vector_store import read_manifest


def _step(label: str, fn) -> bool:
    print(f"- {label} ... ", end="", flush=True)
    try:
        result = fn()
    except Exception as exc:
        print("FAILED")
        print(f"    Reason : {describe_model_error(exc)}")
        print(f"    Raw    : {exc.__class__.__name__}: {str(exc)[:500]}")
        if "--trace" in sys.argv:
            traceback.print_exc()
        return False
    print(f"OK ({result})")
    return True


def main() -> int:
    try:
        s = get_settings()
    except Exception as exc:
        print(f"Could not read settings from .env: {exc}")
        return 1

    key = s.openai_api_key.get_secret_value() if s.openai_api_key else ""
    masked = f"{key[:7]}...{key[-4:]} ({len(key)} chars)" if len(key) > 12 else ("<set>" if key else "<MISSING>")
    print("Configuration")
    print(f"  LLM_PROVIDER       = {s.llm_provider}")
    print(f"  EMBEDDING_PROVIDER = {s.embedding_provider}")
    print(f"  OPENAI_BASE_URL    = {s.openai_base_url or '(default: api.openai.com)'}")
    print(f"  OPENAI_API_KEY     = {masked}")
    if key != key.strip() or key.startswith(("'", '"')):
        print("  WARNING: the API key has quotes or spaces around it - remove them in .env")
    print(f"  EMBEDDING_MODEL    = {s.embedding_model}")
    print(f"  CHAT_MODEL         = {s.chat_model}")
    print()

    ok = _step("Embedding test", lambda: f"vector size {len(get_embeddings(s).embed_query('wheat sowing time'))}")
    ok &= _step("Chat model test", lambda: repr(get_chat_model(s).invoke("Reply with the single word: OK").content[:40]))
    manifest = read_manifest(s.faiss_index_dir)
    print()
    if not manifest:
        print(f"- Knowledge base ... empty ({s.faiss_index_dir})")
    elif manifest.get("embedding_signature") not in (None, s.embedding_signature):
        print("- Knowledge base ... LOCKED")
        print(f"    Built with '{manifest['embedding_signature']}', current setting '{s.embedding_signature}'.")
        print("    Restore the old EMBEDDING_MODEL, or rebuild with:  python -m app.reindex")
        ok = False
    else:
        print(f"- Knowledge base ... OK ({len(manifest.get('sources', {}))} sources)")
    print()
    print("All good - restart the backend and ingest again." if ok else "Fix the FAILED item above, then run this again.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
