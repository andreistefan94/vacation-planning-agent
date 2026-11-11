"""End-to-end flows over the HTTP API with a scripted LLM and a mocked travel API."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

from vacation_agent.agents.conversational import Decision, PreferenceCandidate, TurnAnalysis
from vacation_agent.api.app import ChatService, create_app
from vacation_agent.graph import orchestrator
from vacation_agent.graph.orchestrator import build_graph
from vacation_agent.graph.state import TripRequest


@pytest.fixture
async def api(monkeypatch, travel_api, fake_llm):
    monkeypatch.setattr(orchestrator, "today", lambda: date(2026, 11, 11))
    app = create_app(ChatService(build_graph(MemorySaver())))
    app.state.service = ChatService(build_graph(MemorySaver()))
    travel_api.router.route(host="testserver").pass_through()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


async def chat(client, message, conversation_id=None, user_id="u1"):
    r = await client.post("/chat", json={"user_id": user_id, "conversation_id": conversation_id, "message": message})
    assert r.status_code == 200, r.text
    return r.json()


ITALY = TurnAnalysis(
    trip=TripRequest(
        destination="Italia",
        destination_kind="country",
        candidate_cities_en=["Rome", "Florence", "Atlantis"],
        origin_city_en="Chisinau",
        month=12,
        year=2026,
        duration_days=7,
        adults=2,
        children=1,
        budget=2500,
    ),
    intents=["plan_trip"],
)


async def test_full_trip_with_hitl_choices_and_calendar(api, fake_llm, travel_api):
    fake_llm.analyses = {"Italia": ITALY}
    fake_llm.decisions = {
        "Roma": Decision(
            action="select", option_index=0, trip_changes=TripRequest(start_date="2026-12-18", end_date="2026-12-25")
        ),
        "prima": Decision(action="select", option_index=0),
        "aprob": Decision(action="approve"),
    }
    r1 = await chat(
        api, "Vreau o vacanță de 7 zile în Italia în decembrie, 2 adulți și un copil, buget 2500 EUR, din Chișinău"
    )
    cid = r1["conversation_id"]
    assert r1["status"] == "awaiting_approval"
    assert r1["pending_approval"]["options"] == ["Rome (IT)", "Florence (IT)"]

    r2 = await chat(api, "Roma, plecare pe 18 decembrie și întoarcere pe 25", cid)
    assert r2["status"] == "awaiting_approval"
    assert len(r2["pending_approval"]["options"]) >= 2
    assert "2500.00 EUR" in r2["reply"]  # budget mentioned against real totals

    r3 = await chat(api, "prima variantă", cid)
    assert r3["status"] == "awaiting_approval" and "Program pe zile" in r3["reply"]

    r4 = await chat(api, "Da, aprob și adaug în calendar", cid)
    assert r4["status"] == "completed"
    assert len(travel_api.inserted) == 1 and travel_api.inserted[0]["start"] == {"date": "2026-12-18"}

    history = (await api.get(f"/conversations/{cid}")).json()["messages"]
    users = [m["content"] for m in history if m["role"] == "user"]
    assert users[1].startswith("Roma") and users[-1].startswith("Da, aprob")  # resumed, not restarted
    names = {tc["name"] for m in history for tc in m["tool_calls"]}
    for expected in (
        "get_user_preferences",
        "resolve_location",
        "search_flights",
        "list_hotels_in_city",
        "get_hotel_offers",
        "get_weather_forecast",
        "calculate_budget",
        "add_calendar_event",
    ):
        assert expected in names
    call_ids = {tc["id"] for m in history for tc in m["tool_calls"]}
    result_ids = {m["tool_call_id"] for m in history if m["role"] == "tool"}
    assert call_ids == result_ids
    assert all(m["tool_call_id"] is None for m in history if m["role"] != "tool")


async def test_calendar_not_touched_without_approval(api, fake_llm, travel_api):
    fake_llm.analyses = {"Italia": ITALY}
    fake_llm.decisions = {
        "Roma": Decision(action="select", option_index=0),
        "prima": Decision(action="select", option_index=0),
        "fără": Decision(action="select", option_index=1),
    }
    cid = (await chat(api, "Italia"))["conversation_id"]
    await chat(api, "Roma", cid)
    await chat(api, "prima", cid)
    r = await chat(api, "Aprob, fără calendar", cid)
    assert r["status"] == "completed" and travel_api.inserted == []
    assert "nu au fost rezervate" in r["reply"]


async def test_missing_info_is_a_clarification_not_hitl(api, fake_llm):
    fake_llm.analyses = {
        "Grecia": TurnAnalysis(
            trip=TripRequest(destination="Grecia", destination_kind="country", duration_days=7), intents=["plan_trip"]
        ),
    }
    fake_llm.text = "Pentru a vă propune variante, am nevoie de bugetul aproximativ și orașul de plecare."
    r = await chat(api, "Vreau o săptămână în Grecia")
    assert r["status"] == "completed" and r["pending_approval"] is None
    assert "buget" in r["reply"]


async def test_budget_overrun_requires_approval(api, fake_llm, travel_api):
    travel_api.flight_price = (400.0, 380.0)
    fake_llm.analyses = {
        "Paris": TurnAnalysis(
            trip=TripRequest(
                destination="Paris",
                destination_kind="city",
                destination_city_en="Paris",
                origin_city_en="Chisinau",
                start_date="2027-02-09",
                end_date="2027-02-14",
                adults=2,
                budget=300,
            ),
            intents=["plan_trip"],
        )
    }
    fake_llm.decisions = {"continuă": Decision(action="approve")}
    r1 = await chat(api, "5 zile la Paris, 2 adulți, din Chișinău, buget 300 EUR")
    assert r1["status"] == "awaiting_approval"
    assert "depășesc bugetul" in r1["pending_approval"]["question"]
    r2 = await chat(api, "Da, continuă chiar dacă e peste buget", r1["conversation_id"])
    assert r2["status"] == "awaiting_approval" and "DEPĂȘEȘTE" in r2["reply"]


async def test_preference_saved_only_after_confirmation(api, fake_llm):
    fake_llm.analyses = {
        "mereu": TurnAnalysis(
            trip=TripRequest(),
            intents=["state_preference"],
            preference_candidates=[
                PreferenceCandidate(
                    text="Preferă zboruri directe", category="direct_flights", explicitly_permanent=True
                )
            ],
        ),
        "reține": TurnAnalysis(trip=TripRequest(), intents=["other"], preference_confirmation="yes"),
    }
    r1 = await chat(api, "Prefer mereu zboruri directe", user_id="mem")
    assert "preferință permanentă" in r1["reply"]
    assert (await api.get("/users/mem/preferences")).json()["preferences"] == []
    await chat(api, "Da, reține asta", r1["conversation_id"], user_id="mem")
    assert (await api.get("/users/mem/preferences")).json()["preferences"] == ["Preferă zboruri directe"]

    # a new conversation loads it
    r3 = await chat(api, "Salut", user_id="mem")
    history = (await api.get(f"/conversations/{r3['conversation_id']}")).json()["messages"]
    loaded = next(m for m in history if m["role"] == "tool")
    assert "Preferă zboruri directe" in loaded["content"]

    assert (await api.delete("/users/mem/preferences")).status_code == 200
    assert (await api.get("/users/mem/preferences")).json()["preferences"] == []


async def test_booking_request_is_refused_honestly(api, fake_llm):
    fake_llm.analyses = {
        "Viena": TurnAnalysis(
            trip=TripRequest(destination="Viena", destination_kind="city", destination_city_en="Vienna"),
            intents=["booking_request", "plan_trip"],
        )
    }
    r = await chat(api, "Rezervă-mi acum zborul și hotelul la Viena și confirmă că biletele sunt cumpărate")
    assert "nu pot cumpăra bilete" in r["reply"]
    assert "am rezervat" not in r["reply"].lower()


async def test_no_accommodation_escalates(api, fake_llm, travel_api):
    travel_api.no_hotels = True
    fake_llm.analyses = {
        "Roma": TurnAnalysis(
            trip=TripRequest(
                destination="Roma",
                destination_kind="city",
                destination_city_en="Rome",
                origin_city_en="Chisinau",
                start_date="2027-03-01",
                end_date="2027-03-05",
                adults=1,
                budget=900,
            ),
            intents=["plan_trip"],
        )
    }
    r = await chat(api, "Roma 1-5 martie, 1 adult, din Chișinău, 900 EUR")
    assert r["status"] == "completed" and "nu am găsit cazare" in r["reply"]


async def test_contract_endpoints(api, fake_llm):
    assert (await api.get("/health")).status_code == 200
    assert (await api.get("/conversations/does-not-exist")).status_code == 404
    r = await chat(api, "Salut", user_id="lister")
    convs = (await api.get("/users/lister/conversations")).json()["conversations"]
    assert convs[0]["conversation_id"] == r["conversation_id"] and convs[0]["preview"] == "Salut"
    bad = await api.post("/chat", json={"message": "x"})
    assert bad.status_code == 422


async def test_option_button_click_is_matched_without_llm(api, fake_llm):
    fake_llm.analyses = {"Italia": ITALY}  # no scripted decisions: only the exact-match path can select
    r1 = await chat(api, "Italia")
    option = r1["pending_approval"]["options"][0]
    r2 = await chat(api, option, r1["conversation_id"])  # what the UI sends when a button is clicked
    assert r2["status"] == "awaiting_approval"
    assert r2["pending_approval"]["question"] == "Care variantă vă convine?"


async def test_ui_is_served_at_root(api, tmp_path, monkeypatch):
    from vacation_agent.config import get_settings

    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text("<html>chat</html>", encoding="utf-8")
    monkeypatch.setattr(get_settings(), "ui_dir", tmp_path / "ui")
    r = await api.get("/")
    assert r.status_code == 200 and "chat" in r.text
