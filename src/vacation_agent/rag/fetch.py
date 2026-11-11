"""Download the Wikivoyage guides for the corpus cities as structured JSON.

Usage: ``pdm run fetch-corpus [--out resources/corpus] [--force]``

Writes one ``<city>.json`` per destination::

    {"city": "Rome", "title": "Rome", "url": "https://en.wikivoyage.org/wiki/Rome",
     "license": "CC BY-SA (Wikivoyage)", "revision_id": 123, "retrieved_at": "...",
     "sections": [{"heading": "By plane", "path": "Get in > By plane", "level": 3, "text": "..."}]}

``ingest.py`` reads this format directly (one section = one unit to chunk, ``path`` = citation).
Existing files (e.g. the official course corpus) are never overwritten unless ``--force`` is given.
``--convert-txt`` converts guides previously saved as ``<city>.txt`` + ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from vacation_agent.config import get_settings
from vacation_agent.rag.ingest import LICENSE, WIKIVOYAGE_URLS, parse_sections

log = logging.getLogger(__name__)

API_URL = "https://en.wikivoyage.org/w/api.php"
REQUEST_DELAY_S = 1.0  # be polite: one request per second


def user_agent() -> str:
    """Wikimedia's robot policy (https://w.wiki/4wJS) requires contact info in the User-Agent."""
    contact = get_settings().wikimedia_contact.strip()
    if not contact:
        raise SystemExit(
            "Set WIKIMEDIA_CONTACT in .env (an email address or a URL, e.g. your repository) - "
            "Wikimedia rejects automated requests without contact information (robot policy https://w.wiki/4wJS)."
        )
    return f"vacation-agent-course-project/0.1 ({contact})"


def page_title(url: str) -> str:
    return url.rsplit("/wiki/", 1)[-1]


def file_stem(city: str) -> str:
    return city.lower().replace(" ", "_")


def fetch_page(client: httpx.Client, title: str) -> dict:
    """Plain-text extract of a Wikivoyage page, with section headings in wiki format."""
    response = client.get(
        API_URL,
        params={
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
            "titles": title,
        },
    )
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise RuntimeError(f"Page '{title}' not found or empty")
    return page


def guide_document(
    city: str, url: str, text: str, title: str, revision_id: Any, retrieved_at: str | None = None
) -> dict[str, Any]:
    return {
        "city": city,
        "title": title,
        "url": url,
        "license": LICENSE,
        "revision_id": revision_id,
        "retrieved_at": retrieved_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "sections": parse_sections(text),
    }


def write_guide(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_corpus(out: Path, force: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with httpx.Client(timeout=30, headers={"User-Agent": user_agent()}) as client:
        for i, (city, url) in enumerate(WIKIVOYAGE_URLS.items()):
            target = out / f"{file_stem(city)}.json"
            if target.exists() and not force:
                log.info("Skipping %s (already exists; use --force to overwrite)", target.name)
                continue
            if i:
                time.sleep(REQUEST_DELAY_S)
            page = fetch_page(client, page_title(url))
            doc = guide_document(city, url, page["extract"], page["title"], page.get("lastrevid"))
            write_guide(target, doc)
            written.append(target)
            log.info("Saved %s (%d sections)", target.name, len(doc["sections"]))
    return written


def convert_txt(out: Path) -> list[Path]:
    """Convert <city>.txt + metadata.json (older fetch format) to <city>.json, then remove them."""
    meta_path = out / "metadata.json"
    metadata = {m["file"]: m for m in json.loads(meta_path.read_text(encoding="utf-8"))} if meta_path.exists() else {}
    written: list[Path] = []
    for city, url in WIKIVOYAGE_URLS.items():
        txt = out / f"{file_stem(city)}.txt"
        if not txt.exists():
            continue
        m = metadata.get(txt.name, {})
        doc = guide_document(
            city,
            m.get("url", url),
            txt.read_text(encoding="utf-8"),
            m.get("title", city),
            m.get("revision_id"),
            m.get("retrieved_at"),
        )
        target = txt.with_suffix(".json")
        write_guide(target, doc)
        txt.unlink()
        written.append(target)
        log.info("Converted %s -> %s (%d sections)", txt.name, target.name, len(doc["sections"]))
    if meta_path.exists() and written:
        meta_path.unlink()
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=get_settings().corpus_dir)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--convert-txt", action="store_true", help="Convert existing .txt guides to JSON.")
    args = parser.parse_args()
    written = convert_txt(args.out) if args.convert_txt else fetch_corpus(args.out, force=args.force)
    print(f"{len(written)} guide file(s) written to {args.out}")


if __name__ == "__main__":
    main()
