#!/bin/sh
set -e
# The RAG index is normally built into the image; rebuild it if DATA_DIR was mounted empty.
if [ ! -d "$DATA_DIR/chroma" ] && [ -d "$CORPUS_DIR" ]; then
  python -m vacation_agent.rag.ingest --corpus "$CORPUS_DIR" || echo "RAG ingestion failed; continuing without guides"
fi
exec uvicorn vacation_agent.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
