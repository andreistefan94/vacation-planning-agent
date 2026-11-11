from langchain_core.embeddings import DeterministicFakeEmbedding

from vacation_agent.config import get_settings
from vacation_agent.rag import retriever
from vacation_agent.rag.ingest import chunk_documents, load_corpus, split_sections

GUIDE = (
    """Rome is the capital of Italy.

== Get in ==
Fiumicino airport is linked to Termini station by the Leonardo Express train every 15 minutes.
"""
    + "More details about trains and buses in the city. " * 10
    + """

== Districts ==
Trastevere is a lively district with narrow streets and many restaurants.
"""
    + "It is popular with visitors in the evening. " * 10
)


def test_split_sections_on_wiki_headings():
    sections = dict(split_sections(GUIDE))
    assert {"Overview", "Get in", "Districts"} <= set(sections)


async def test_ingest_and_filtered_search_with_citations(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rome.txt").write_text(GUIDE, encoding="utf-8")
    (corpus / "paris.md").write_text(
        "# Paris\n\n## Get around\n" + "The metro covers the whole city. " * 10, encoding="utf-8"
    )
    docs = load_corpus(corpus)
    assert {d.city for d in docs} == {"Rome", "Paris"}
    chunks = chunk_documents(docs)
    assert all(c.metadata["license"].startswith("CC BY-SA") and c.metadata["source_url"] for c in chunks)

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="test_guides",
        embedding_function=DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks)
    retriever.set_vectorstore(store)
    monkeypatch.setattr(retriever, "index_size", lambda: len(chunks))
    monkeypatch.setattr(get_settings(), "rag_max_distance", 2.0)

    res = await retriever.search_travel_guide("how to get from the airport", city="Roma")
    assert res["ok"] and res["results"]
    assert all(r["city"] == "Rome" for r in res["results"])  # metadata filter applied
    assert "wikivoyage.org/wiki/Rome" in retriever.format_citation(res["results"][0])


def test_json_guides_from_fetch_are_ingested_with_section_paths(tmp_path):
    from vacation_agent.rag.fetch import guide_document, write_guide

    doc = guide_document(
        "Rome", "https://en.wikivoyage.org/wiki/Rome", GUIDE + "\n=== By bus ===\n" + "Buses. " * 30, "Rome", 1
    )
    assert [s["path"] for s in doc["sections"]] == ["Overview", "Get in", "Districts", "Districts > By bus"]
    write_guide(tmp_path / "rome.json", doc)
    docs = load_corpus(tmp_path)
    assert docs[0].city == "Rome" and docs[0].sections
    sections = {c.metadata["section"] for c in chunk_documents(docs)}
    assert "Districts > By bus" in sections


async def test_city_outside_corpus_is_recognised():
    res = await retriever.search_travel_guide("what to see", city="Athens")
    assert res["ok"] and res["in_corpus"] is False and res["results"] == []
