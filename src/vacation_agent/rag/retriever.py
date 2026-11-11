"""Semantic search over the Wikivoyage corpus, with metadata (city) filtering and citations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vacation_agent.config import get_settings
from vacation_agent.rag.ingest import WIKIVOYAGE_URLS, canonical_city

log = logging.getLogger(__name__)

_vectorstore = None

CORPUS_CITIES = list(WIKIVOYAGE_URLS)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        from vacation_agent.rag.embeddings import get_embeddings

        s = get_settings()
        _vectorstore = Chroma(
            collection_name=s.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(s.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore


def set_vectorstore(store) -> None:
    global _vectorstore
    _vectorstore = store


def reset_vectorstore() -> None:
    store = get_vectorstore()
    store.delete_collection()
    set_vectorstore(None)


def index_size() -> int:
    try:
        return get_vectorstore()._collection.count()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return 0


class GuideSearchInput(BaseModel):
    query: str = Field(description="What to look up, in English (e.g. 'getting from the airport to the centre').")
    city: str | None = Field(
        default=None, description=f"Restrict to one destination. Corpus covers: {', '.join(CORPUS_CITIES)}."
    )


async def search_travel_guide(query: str, city: str | None = None) -> dict[str, Any]:
    """Search the Wikivoyage travel guides (static info: neighbourhoods, getting around, entry rules, safety).

    Never use it for prices, availability, weather or opening hours.
    """
    canonical = canonical_city(city) if city else None
    if city and not canonical:
        return {
            "ok": True,
            "results": [],
            "in_corpus": False,
            "note": f"'{city}' is not covered by the guide corpus ({', '.join(CORPUS_CITIES)}).",
        }
    if index_size() == 0:
        return {"ok": False, "error": "The travel guide index is empty (run `pdm run ingest`)."}
    s = get_settings()
    try:
        store = get_vectorstore()
        hits = await store.asimilarity_search_with_score(
            query, k=s.rag_top_k, filter={"city": canonical} if canonical else None
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Guide search failed: {exc}"}
    results = []
    for doc, distance in hits:
        if distance > s.rag_max_distance:
            continue
        md = doc.metadata
        results.append(
            {
                "text": doc.page_content,
                "city": md.get("city"),
                "section": md.get("section"),
                "source_url": md.get("source_url"),
                "license": md.get("license", "CC BY-SA"),
                "distance": round(float(distance), 3),
            }
        )
    return {
        "ok": True,
        "query": query,
        "city": canonical,
        "in_corpus": True,
        "results": results,
        "note": "" if results else "No sufficiently relevant passage in the guides.",
    }


def format_citation(result: dict[str, Any]) -> str:
    return (
        f"Wikivoyage – {result.get('city')}, secțiunea „{result.get('section')}” "
        f"(licență {result.get('license', 'CC BY-SA').replace(' (Wikivoyage)', '')}): {result.get('source_url')}"
    )


guide_tool = StructuredTool.from_function(
    coroutine=search_travel_guide,
    name="search_travel_guide",
    description=(search_travel_guide.__doc__ or "").strip(),
    args_schema=GuideSearchInput,
)
RAG_TOOLS = {guide_tool.name: guide_tool}
