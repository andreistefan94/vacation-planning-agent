"""Index the Wikivoyage corpus into Chroma.

Usage: ``pdm run ingest [--corpus resources/corpus] [--reset]``

The loader is tolerant to the corpus format: it accepts .md/.txt/.html files (city inferred from a
metadata file or from the file name) and .json/.jsonl files with text + metadata fields.
Chunks are split along the guide's section headings first, then by size, and every chunk keeps
``city``, ``section``, ``source_url``, ``title`` and ``license`` metadata for filtering and citation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vacation_agent.config import get_settings

log = logging.getLogger(__name__)

KNOWN_CITIES = {
    "paris": "Paris",
    "rome": "Rome",
    "roma": "Rome",
    "barcelona": "Barcelona",
    "london": "London",
    "londra": "London",
    "new_york": "New York City",
    "new york": "New York City",
    "new-york": "New York City",
    "nyc": "New York City",
    "tokyo": "Tokyo",
    "bangkok": "Bangkok",
    "dubai": "Dubai",
    "sydney": "Sydney",
    "cairo": "Cairo",
}
WIKIVOYAGE_URLS = {
    "Paris": "https://en.wikivoyage.org/wiki/Paris",
    "Rome": "https://en.wikivoyage.org/wiki/Rome",
    "Barcelona": "https://en.wikivoyage.org/wiki/Barcelona",
    "London": "https://en.wikivoyage.org/wiki/London",
    "New York City": "https://en.wikivoyage.org/wiki/New_York_City",
    "Tokyo": "https://en.wikivoyage.org/wiki/Tokyo",
    "Bangkok": "https://en.wikivoyage.org/wiki/Bangkok",
    "Dubai": "https://en.wikivoyage.org/wiki/Dubai",
    "Sydney": "https://en.wikivoyage.org/wiki/Sydney",
    "Cairo": "https://en.wikivoyage.org/wiki/Cairo",
}
LICENSE = "CC BY-SA (Wikivoyage)"
TEXT_KEYS = ("text", "content", "body", "page_content")
CITY_KEYS = ("city", "destination", "location", "name")
URL_KEYS = ("url", "source", "source_url", "link")

_HEADING = re.compile(r"^(#{1,6}\s+.+|={2,6}\s*[^=]+?\s*={2,6})\s*$", re.MULTILINE)


@dataclass
class RawDoc:
    text: str
    city: str
    url: str
    title: str
    sections: list[tuple[str, str]] | None = None  # pre-split (section path, body), e.g. from the JSON corpus
    revision_id: int = 0  # Wikivoyage revision the text comes from (0 = unknown)


def permalink(url: str, revision_id: int | None) -> str:
    """Link to the exact revision that was indexed (stable even if the page changes or is renamed)."""
    if not revision_id or "/wiki/" not in url:
        return url
    base, title = url.split("/wiki/", 1)
    return f"{base}/w/index.php?title={title}&oldid={revision_id}"


def canonical_city(value: str) -> str | None:
    key = value.strip().lower().replace(".md", "").replace(".txt", "")
    if key in KNOWN_CITIES:
        return KNOWN_CITIES[key]
    for k, v in KNOWN_CITIES.items():
        if k in key:
            return v
    return None


def _strip_html(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(
        r"<h([1-6])[^>]*>(.*?)</h\1>",
        lambda m: "\n" + "#" * int(m.group(1)) + " " + m.group(2) + "\n",
        text,
        flags=re.S | re.I,
    )
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"[ \t]+", " ", text))


def _load_metadata(corpus: Path) -> dict[str, dict]:
    """Optional metadata file mapping file name -> {city, url}."""
    meta: dict[str, dict] = {}
    for path in corpus.rglob("*"):
        if path.name.lower().startswith(("metadata", "manifest", "index")) and path.suffix in (".json", ".csv"):
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else data.get("documents", data)
                if isinstance(items, dict):
                    items = [{"file": k, **(v if isinstance(v, dict) else {"city": v})} for k, v in items.items()]
            else:
                with path.open(encoding="utf-8") as fh:
                    items = list(csv.DictReader(fh))
            for item in items:
                fname = item.get("file") or item.get("filename") or item.get("path") or item.get("id")
                if fname:
                    meta[Path(str(fname)).name] = item
    return meta


def _first(d: dict, keys: tuple[str, ...], default: str = "") -> str:
    for k in keys:
        if d.get(k):
            return str(d[k])
    return default


def load_corpus(corpus: Path) -> list[RawDoc]:
    meta = _load_metadata(corpus)
    docs: list[RawDoc] = []
    for path in sorted(corpus.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower().startswith(("metadata", "manifest", "index")):
            continue
        suffix = path.suffix.lower()
        if suffix in (".json", ".jsonl"):
            raw = path.read_text(encoding="utf-8")
            records = (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if suffix == ".jsonl"
                else json.loads(raw)
            )
            if isinstance(records, dict):
                records = records.get("documents") or records.get("data") or [records]
            for rec in records:
                rec_meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
                merged = {**rec, **rec_meta}
                city = canonical_city(_first(merged, CITY_KEYS)) or canonical_city(path.stem)
                if not city:
                    log.warning("Skipping a record in %s: cannot infer the city", path)
                    continue
                url = _first(merged, URL_KEYS, WIKIVOYAGE_URLS.get(city, ""))
                title = _first(merged, ("title",), city)
                if isinstance(rec.get("sections"), list):  # guide JSON written by rag/fetch.py
                    sections = [
                        (s.get("path") or s.get("heading") or "Overview", s.get("text", ""))
                        for s in rec["sections"]
                        if s.get("text", "").strip()
                    ]
                    revision = int(merged.get("revision_id") or 0)
                    docs.append(RawDoc("", city, url, title, sections=sections, revision_id=revision))
                elif text := _first(rec, TEXT_KEYS):
                    docs.append(RawDoc(text, city, url, title, revision_id=int(merged.get("revision_id") or 0)))
        elif suffix in (".md", ".txt", ".html", ".htm"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if suffix in (".html", ".htm"):
                text = _strip_html(text)
            m = meta.get(path.name, {})
            city = canonical_city(_first(m, CITY_KEYS)) if m else None
            city = city or canonical_city(path.stem) or canonical_city(path.parent.name)
            if not city:
                log.warning("Skipping %s: cannot infer the city", path)
                continue
            url = _first(m, URL_KEYS, WIKIVOYAGE_URLS.get(city, ""))
            docs.append(RawDoc(text, city, url, _first(m, ("title",), city)))
    return docs


def parse_sections(text: str) -> list[dict]:
    """Split a guide on markdown/wiki headings, keeping the heading level and full path.

    Returns ``[{"heading", "path", "level", "text"}]``; ``path`` joins the parent headings,
    e.g. ``"Get in > By plane"``. Text before the first heading is the ``Overview`` section.
    """
    matches = list(_HEADING.finditer(text))
    sections: list[dict] = []
    intro = text[: matches[0].start()] if matches else text
    if intro.strip():
        sections.append({"heading": "Overview", "path": "Overview", "level": 1, "text": intro.strip()})
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        marker = m.group(1).strip()
        level = len(marker) - len(marker.lstrip("#=")) if marker[0] in "#=" else 1
        if marker.startswith("#"):
            level += 1  # "# Title" is the page title in markdown; "##" ~ "=="
        heading = marker.strip("#= ").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            path = " > ".join(h for _, h in stack)
            sections.append({"heading": heading, "path": path, "level": level, "text": body})
    return sections


def split_sections(text: str) -> list[tuple[str, str]]:
    """(section path, body) pairs of a plain-text/markdown guide."""
    return [(s["path"], s["text"]) for s in parse_sections(text)]


def chunk_documents(docs: list[RawDoc], chunk_size: int = 1500, overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Document] = []
    for doc in docs:
        for section, body in doc.sections or split_sections(doc.text):
            for i, piece in enumerate(splitter.split_text(body)):
                if len(piece) < 80:
                    continue
                chunks.append(
                    Document(
                        page_content=f"{doc.city} — {section}\n{piece}",
                        metadata={
                            "city": doc.city,
                            "section": section[:120],
                            "source_url": permalink(doc.url, doc.revision_id),
                            "page_url": doc.url,
                            "revision_id": doc.revision_id,
                            "title": doc.title,
                            "license": LICENSE,
                            "chunk": i,
                        },
                    )
                )
    return chunks


def chunk_id(city: str, revision_id: int, n: int) -> str:
    return f"{city}|rev{revision_id}|{n}"


def index_documents(docs: list[RawDoc], batch: int = 64) -> int:
    """Add the documents' chunks, then drop older chunks of the same cities.

    New chunks are added before the old ones are deleted, so searches never see a city without
    content while it is being re-indexed.
    """
    from vacation_agent.rag.retriever import get_vectorstore

    store = get_vectorstore()
    total = 0
    for doc in docs:
        chunks = chunk_documents([doc])
        ids = [chunk_id(doc.city, doc.revision_id, n) for n in range(len(chunks))]
        for i in range(0, len(chunks), batch):
            store.add_documents(chunks[i : i + batch], ids=ids[i : i + batch])
        keep = set(ids)
        existing = store.get(where={"city": doc.city}, include=[])["ids"]
        stale = [i for i in existing if i not in keep]
        if stale:
            store.delete(ids=stale)
        total += len(chunks)
        log.info(
            "Indexed %s: %d chunks (revision %s), removed %d old", doc.city, len(chunks), doc.revision_id, len(stale)
        )
    return total


def build_index(corpus: Path, reset: bool = False, cities: set[str] | None = None) -> int:
    from vacation_agent.rag.retriever import reset_vectorstore

    docs = [d for d in load_corpus(corpus) if cities is None or d.city in cities]
    if not docs:
        raise SystemExit(f"No documents found in {corpus}")
    if reset:
        reset_vectorstore()
    return index_documents(docs)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=get_settings().corpus_dir)
    parser.add_argument("--reset", action="store_true", help="Drop the existing collection first.")
    args = parser.parse_args()
    n = build_index(args.corpus, reset=args.reset)
    print(f"Indexed {n} chunks from {args.corpus}")


if __name__ == "__main__":
    main()
