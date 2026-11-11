"""Embedding models for RAG: local sentence-transformers (default) or Gemini."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from vacation_agent.config import get_settings

log = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain wrapper over a local sentence-transformers model (runs on CPU, no API calls).

    Vectors are L2-normalized, so cosine distance in Chroma is well-behaved.
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embeddings_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=s.gemini_embedding_model, google_api_key=s.google_api_key or None)
    return SentenceTransformerEmbeddings(s.local_embedding_model)
