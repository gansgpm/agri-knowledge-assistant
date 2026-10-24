# Agriculture Knowledge Assistant
# One image runs both services; docker-compose.yml picks the command for each.
#   backend : uvicorn (FastAPI)  -> port 8000
#   frontend: streamlit          -> port 8501
# Secrets are never baked in: pass .env at run time (docker compose does this).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code changes don't re-download packages.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code.
COPY app ./app
COPY ui ./ui
COPY tests ./tests
COPY pytest.ini .env.example README.md ./

# Run as a non-root user; data/ holds the FAISS index, uploads and embedding cache
# and is mounted as a volume so it survives container restarts.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default command: the backend. The frontend overrides it in docker-compose.yml.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
