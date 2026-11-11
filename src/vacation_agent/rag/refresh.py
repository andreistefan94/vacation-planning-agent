"""Periodic corpus refresh: keep the Wikivoyage guides (and the RAG index) up to date automatically.

Every ``CORPUS_REFRESH_DAYS`` the API server:
1. asks Wikivoyage, in ONE request, for the latest revision id of the 10 guide pages;
2. downloads only the guides whose revision differs from the local ``<city>.json``;
3. re-indexes only those cities (new chunks are added before the old ones are removed).

The state (last run, revisions, what changed) is kept in ``DATA_DIR/corpus_refresh.json``, so a
restart does not trigger an extra refresh. Manual run: ``pdm run refresh-corpus [--force]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from vacation_agent.config import get_settings
from vacation_agent.rag.fetch import API_URL, fetch_page, file_stem, guide_document, page_title, user_agent
from vacation_agent.rag.ingest import WIKIVOYAGE_URLS, build_index

log = logging.getLogger(__name__)

RETRY_AFTER_FAILURE = timedelta(hours=6)


def state_path() -> Path:
    return get_settings().data_dir / "corpus_refresh.json"


def load_state() -> dict[str, Any]:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_revisions(client: httpx.Client) -> dict[str, int]:
    """Current revision id of every guide page, in a single API call (follows renames/redirects)."""
    titles = {city: page_title(url) for city, url in WIKIVOYAGE_URLS.items()}
    response = client.get(
        API_URL,
        params={
            "action": "query",
            "prop": "info",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
            "titles": "|".join(titles.values()),
        },
    )
    response.raise_for_status()
    query = response.json()["query"]
    mapping = {t: t.replace("_", " ") for t in titles.values()}
    for item in query.get("normalized", []):
        mapping = {k: (item["to"] if v == item["from"] else v) for k, v in mapping.items()}
    for item in query.get("redirects", []):
        mapping = {k: (item["to"] if v == item["from"] else v) for k, v in mapping.items()}
    by_title = {p["title"]: p.get("lastrevid") for p in query["pages"] if not p.get("missing")}
    return {city: by_title[mapping[t]] for city, t in titles.items() if by_title.get(mapping[t])}


def local_revisions(corpus: Path) -> dict[str, int]:
    revisions: dict[str, int] = {}
    for city in WIKIVOYAGE_URLS:
        path = corpus / f"{file_stem(city)}.json"
        try:
            revisions[city] = int(json.loads(path.read_text(encoding="utf-8")).get("revision_id") or 0)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
    return revisions


def _write_atomic(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def refresh_corpus(corpus: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Download changed guides and re-index them. Returns the new state."""
    corpus = corpus or get_settings().corpus_dir
    corpus.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, headers={"User-Agent": user_agent()}) as client:
        remote = latest_revisions(client)
        local = local_revisions(corpus)
        changed = [c for c, rev in remote.items() if force or local.get(c) != rev]
        for city in changed:
            url = WIKIVOYAGE_URLS[city]
            page = fetch_page(client, page_title(url))
            doc = guide_document(city, url, page["extract"], page["title"], page.get("lastrevid"))
            _write_atomic(corpus / f"{file_stem(city)}.json", doc)
            log.info("Corpus refresh: %s updated (revision %s -> %s)", city, local.get(city), page.get("lastrevid"))
    if changed:
        build_index(corpus, cities=set(changed))
    state = {
        "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "ok",
        "changed": changed,
        "revisions": {**local, **remote},
    }
    save_state(state)
    log.info("Corpus refresh done: %d of %d guides changed", len(changed), len(remote))
    return state


async def refresh_loop() -> None:
    """Background task started by the API server; runs ``refresh_corpus`` every CORPUS_REFRESH_DAYS."""
    settings = get_settings()
    if settings.corpus_refresh_days <= 0:
        log.info("Corpus refresh disabled (CORPUS_REFRESH_DAYS <= 0)")
        return
    if not settings.wikimedia_contact.strip():
        log.warning("Corpus refresh disabled: set WIKIMEDIA_CONTACT (required by Wikimedia's robot policy)")
        return
    interval = timedelta(days=settings.corpus_refresh_days)
    while True:
        state = load_state()
        last = datetime.fromisoformat(state["last_run"]) if state.get("last_run") else None
        wait = interval if state.get("status") == "ok" else RETRY_AFTER_FAILURE
        due = (last + wait) if last else datetime.now(UTC)
        delay = max(0.0, (due - datetime.now(UTC)).total_seconds())
        log.info(
            "Next corpus refresh at %s", (datetime.now(UTC) + timedelta(seconds=delay)).isoformat(timespec="seconds")
        )
        await asyncio.sleep(delay)
        try:
            await asyncio.to_thread(refresh_corpus)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never let the refresh take the API down
            log.exception("Corpus refresh failed; keeping the current corpus")
            save_state(
                {
                    **load_state(),
                    "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                    "status": "failed",
                    "error": str(exc),
                }
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-index every guide.")
    args = parser.parse_args()
    state = refresh_corpus(force=args.force)
    print(f"Changed: {', '.join(state['changed']) or 'none'}")


if __name__ == "__main__":
    main()
