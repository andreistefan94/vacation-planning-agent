FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PDM_CHECK_UPDATE=false \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

RUN pip install --no-cache-dir pdm

COPY pyproject.toml pdm.lock README.md ./
COPY src ./src
RUN pdm install --prod --frozen-lockfile --no-editable

ENV DATA_DIR=/app/data CORPUS_DIR=/app/resources/corpus UI_DIR=/app/resources/ui PATH="/app/.venv/bin:$PATH"

# Bake the embedding model and the RAG index into the image (local embeddings: no API key needed).
COPY resources ./resources
RUN python -m vacation_agent.rag.ingest --corpus "$CORPUS_DIR"

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
