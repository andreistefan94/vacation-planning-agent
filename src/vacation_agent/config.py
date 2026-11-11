"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    llm_temperature: float = 0.2

    # Mock travel APIs (primary + fallback mirror)
    mock_api_base_url: str = "https://lab-10000-6-3036-svc50-27zuw5t5ra-lm.a.run.app"
    mock_api_fallback_url: str = "https://mock-apis.mhyc5y7etwse6.eu-central-1.cs.amazonlightsail.com"
    mock_client_id: str = "vacation-agent"
    mock_client_secret: str = "vacation-agent-secret"
    http_timeout_s: float = 20.0
    http_max_attempts: int = 3

    # "mock" (evaluation) or "real" (Open-Meteo for weather; other tools stay mock)
    tools_backend: str = "mock"

    # Calendar used for checking availability / inserting the trip
    calendar_id: str = "primary"

    # Persistence
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("resources/corpus")
    ui_dir: Path = Path("resources/ui")  # course-provided chat UI, served at "/"

    # RAG
    wikimedia_contact: str = ""  # email or URL sent in the User-Agent when downloading the corpus
    corpus_refresh_days: float = 7.0  # background refresh of the guides; 0 disables it
    embeddings_provider: str = "local"  # "local" (sentence-transformers) or "gemini"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_collection: str = "wikivoyage"
    rag_top_k: int = 4
    rag_max_distance: float = 0.68  # cosine distance; calibrated on the corpus (relevant <= 0.67, unrelated >= 0.70)

    # Domain defaults
    default_currency: str = "EUR"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def embedding_model(self) -> str:
        return self.local_embedding_model if self.embeddings_provider == "local" else self.gemini_embedding_model

    @property
    def collection_name(self) -> str:
        """One collection per embedding model: vectors from different models are not comparable."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.embedding_model.split("/")[-1]).strip("-").lower()
        return f"{self.rag_collection}-{slug}"[:500]


@lru_cache
def get_settings() -> Settings:
    # Export .env into os.environ too: libraries configured only through environment variables
    # (LangSmith tracing: LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT) read it from there.
    # Variables already set in the real environment win.
    load_dotenv(override=False)
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
