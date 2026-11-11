import json

import httpx
import respx

from vacation_agent.config import get_settings
from vacation_agent.rag import refresh
from vacation_agent.rag.fetch import API_URL, guide_document, write_guide
from vacation_agent.rag.ingest import WIKIVOYAGE_URLS, permalink


def test_permalink_points_to_the_indexed_revision():
    assert (
        permalink("https://en.wikivoyage.org/wiki/New_York_City", 42)
        == "https://en.wikivoyage.org/w/index.php?title=New_York_City&oldid=42"
    )
    assert permalink("https://en.wikivoyage.org/wiki/Rome", 0) == "https://en.wikivoyage.org/wiki/Rome"


def _info_response(revisions: dict[str, int]) -> dict:
    return {
        "query": {
            "normalized": [{"from": "New_York_City", "to": "New York City"}],
            "redirects": [{"from": "Cairo", "to": "Cairo (Egypt)"}],
            "pages": [{"title": "Cairo (Egypt)" if c == "Cairo" else c, "lastrevid": r} for c, r in revisions.items()],
        }
    }


def test_refresh_downloads_and_reindexes_only_changed_guides(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "wikimedia_contact", "https://example.org/project")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    remote = {city: 100 for city in WIKIVOYAGE_URLS}
    for city, url in WIKIVOYAGE_URLS.items():  # local copies at the same revision...
        write_guide(
            corpus / f"{city.lower().replace(' ', '_')}.json", guide_document(city, url, "Old text.", city, 100)
        )
    remote["Rome"] = 101  # ...except Rome, which changed online
    remote["Cairo"] = 100  # renamed page (redirect) but unchanged revision

    reindexed: list[set] = []
    monkeypatch.setattr(refresh, "build_index", lambda corpus, cities: reindexed.append(cities))

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("prop") == "info":
            return httpx.Response(200, json=_info_response(remote))
        title = request.url.params["titles"]
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [{"title": title, "lastrevid": 101, "extract": "New text.\n== Get in ==\nBy plane."}]
                }
            },
        )

    with respx.mock() as router:
        fetches = router.get(API_URL).mock(side_effect=api)
        state = refresh.refresh_corpus(corpus)

    assert state["changed"] == ["Rome"] and state["status"] == "ok"
    assert reindexed == [{"Rome"}]
    assert fetches.call_count == 2  # one revision check for all pages + one download
    rome = json.loads((corpus / "rome.json").read_text(encoding="utf-8"))
    assert rome["revision_id"] == 101 and rome["sections"][-1]["path"] == "Get in"
    assert refresh.load_state()["revisions"]["Rome"] == 101


async def test_refresh_loop_is_disabled_without_contact(monkeypatch):
    monkeypatch.setattr(get_settings(), "wikimedia_contact", "")
    await refresh.refresh_loop()  # returns immediately instead of scheduling
