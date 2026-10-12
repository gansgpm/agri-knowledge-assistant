"""Turn model/provider exceptions into short, actionable messages for the UI."""

from __future__ import annotations


def describe_model_error(exc: BaseException) -> str:
    """Explain the most likely cause of an embedding / chat-model failure.

    Works by class name and message text so it needs no provider imports and
    keeps working if another LangChain provider is plugged in.
    """
    name = exc.__class__.__name__
    text = str(exc)
    low = text.lower()

    if "openai_api_key" in low or "api_key client option must be set" in low:
        return "No API key configured. Set OPENAI_API_KEY in .env and restart the backend."
    if name == "AuthenticationError" or "401" in text or "invalid_api_key" in low or "incorrect api key" in low:
        return "The API key was rejected (401). Check OPENAI_API_KEY in .env (no quotes or spaces) and restart the backend."
    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return ("Your OpenAI account has no credit left (insufficient_quota). Add billing credit at "
                "platform.openai.com/settings/organization/billing, or switch to Ollama (free).")
    if name == "RateLimitError" or "429" in text:
        return "Rate limit hit (429). Wait a minute and retry, or ingest fewer/smaller files at once."
    if name == "NotFoundError" or "model_not_found" in low or "does not exist" in low or (
        "model" in low and "not found" in low
    ):
        return ("The model name was not found. Check EMBEDDING_MODEL / CHAT_MODEL in .env "
                "(for Ollama, run `ollama pull <model>` first).")
    if "certificate" in low or "ssl" in low:
        return ("SSL certificate error - usually a corporate network/VPN/proxy intercepting HTTPS. "
                "Try another network, or ask IT for the proxy CA certificate (set REQUESTS_CA_BUNDLE and SSL_CERT_FILE).")
    if "tiktoken" in low or "openaipublic" in low:
        return ("Could not download the tokenizer file (tiktoken). Your network may block "
                "openaipublic.blob.core.windows.net - try another network.")
    if name in ("APIConnectionError", "ConnectError", "ConnectionError", "ConnectionRefusedError") or "connection" in low:
        return ("Could not connect to the model server. If using OpenAI, check your internet/proxy; "
                "if using Ollama, make sure Ollama is running and OPENAI_BASE_URL=http://localhost:11434/v1.")
    if "timeout" in low or "timed out" in low:
        return "The model server timed out. Retry, or raise LLM_TIMEOUT_SECONDS in .env."
    if name == "BadRequestError" or "400" in text:
        return f"The model server rejected the request (400): {text[:200]}"
    return f"{name}: {text[:200]}"
