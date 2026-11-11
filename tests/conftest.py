"""Shared fixtures: isolated settings/data dir, a scripted fake LLM and a mocked travel API."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
import respx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

BASE = "https://mock.test"
FALLBACK = "https://mock-fallback.test"

os.environ["MOCK_API_BASE_URL"] = BASE
os.environ["MOCK_API_FALLBACK_URL"] = FALLBACK
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith (set before .env is loaded)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["HTTP_MAX_ATTEMPTS"] = "2"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from vacation_agent import config
    from vacation_agent.llm import set_llm
    from vacation_agent.memory import store as store_mod
    from vacation_agent.rag import retriever
    from vacation_agent.tools import client as client_mod

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    store_mod.set_store(None)
    client_mod.set_client(None)
    retriever.set_vectorstore(None)
    monkeypatch.setattr(retriever, "index_size", lambda: 0)
    yield
    set_llm(None)
    store_mod.set_store(None)
    client_mod.set_client(None)
    config.get_settings.cache_clear()


# ------------------------------------------------------------------ fake LLM


class FakeLLM(BaseChatModel):
    """Structured outputs are scripted by substring of the last message; free text is canned."""

    analyses: dict[str, Any] = Field(default_factory=dict)
    decisions: dict[str, Any] = Field(default_factory=dict)
    text: str = "- Ziua 1: sosire și plimbare prin centru.\n- Ziua 2: muzee."

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    def bind_tools(self, tools, **kwargs):  # no tool calls -> specialists use their deterministic plans
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(messages):
            from vacation_agent.agents.conversational import Decision, TurnAnalysis, heuristic_decision
            from vacation_agent.graph.state import TripRequest

            last = messages[-1].content
            if schema is TurnAnalysis:
                latest = last.split("Latest user message:")[-1]
                for key, value in self.analyses.items():
                    if key in latest:
                        return value
                return TurnAnalysis(trip=TripRequest(), intents=["other"])
            if schema is Decision:
                for key, value in self.decisions.items():
                    if key in last:
                        return value
                return heuristic_decision(last, []) or Decision(action="unclear")
            raise AssertionError(f"unexpected schema {schema}")

        return RunnableLambda(run)


@pytest.fixture
def fake_llm():
    from vacation_agent.llm import set_llm

    llm = FakeLLM()
    set_llm(llm)
    return llm


# ------------------------------------------------------------------ mocked travel API

LOCATIONS = {
    "rome": ("ROM", "Rome", "IT", 41.9, 12.5),
    "chisinau": ("KIV", "Chisinau", "MD", 47.0, 28.86),
    "paris": ("PAR", "Paris", "FR", 48.85, 2.35),
    "florence": ("FLR", "Florence", "IT", 43.77, 11.25),
    "kiv": ("KIV", "Chisinau", "MD", 47.0, 28.86),
}


def _location(code, city, country, lat, lon):
    return {
        "type": "location",
        "subType": "CITY",
        "iataCode": code,
        "name": city.upper(),
        "address": {"cityName": city, "cityCode": code, "countryCode": country},
        "geoCode": {"latitude": lat, "longitude": lon},
    }


def _flight(idx, origin, dest, day, price, stops=0, duration="PT2H30M"):
    return {
        "type": "flight-offer",
        "id": str(idx),
        "numberOfBookableSeats": 5,
        "itineraries": [
            {
                "duration": duration,
                "segments": [
                    {
                        "departure": {"iataCode": origin, "at": f"{day}T09:00:00"},
                        "arrival": {"iataCode": dest, "at": f"{day}T11:30:00"},
                        "carrierCode": "XX",
                        "number": str(100 + idx),
                        "duration": duration,
                        "numberOfStops": stops,
                    }
                ],
            }
        ],
        "price": {"currency": "EUR", "total": str(price), "base": str(price), "grandTotal": str(price)},
        "travelerPricings": [{"travelerId": "1", "price": {"currency": "EUR", "total": str(price)}}],
    }


HOTELS = [
    ("H1", "Budget Inn", "STANDARD", 50.0),
    ("H2", "City Hotel", "SUPERIOR", 90.0),
    ("H3", "Grand Suite", "SUITE", 200.0),
]


class MockTravelApi:
    def __init__(self, router: respx.MockRouter):
        self.router = router
        self.flight_price = (150.0, 120.0)
        self.flight_stops = 0
        self.calendar_events: list[dict] = []
        self.inserted: list[dict] = []
        self.fail_flights_times = 0
        self.no_hotels = False

    def install(self):
        r = self.router
        r.post(f"{BASE}/security/oauth2/token").mock(
            return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 1799, "token_type": "Bearer"})
        )
        r.get(f"{BASE}/reference-data/locations").mock(side_effect=self._locations)
        r.get(f"{BASE}/shopping/flight-offers").mock(side_effect=self._flights)
        r.get(f"{BASE}/reference-data/locations/hotels/by-city").mock(side_effect=self._hotels)
        r.get(f"{BASE}/shopping/hotel-offers").mock(side_effect=self._offers)
        r.get(f"{BASE}/v1/forecast").mock(side_effect=self._forecast)
        r.post(f"{BASE}/shopping/flight-offers").mock(side_effect=self._flights_post)
        r.get(url__regex=rf"{BASE}/calendar/v3/calendars/.+/events").mock(
            side_effect=lambda request: httpx.Response(200, json={"items": self.calendar_events})
        )
        r.post(url__regex=rf"{BASE}/calendar/v3/calendars/.+/events").mock(side_effect=self._insert)
        r.route(host="mock-fallback.test").mock(return_value=httpx.Response(503))
        return self

    def _locations(self, request):
        kw = request.url.params.get("keyword", "").lower()
        data = [_location(*LOCATIONS[kw])] if kw in LOCATIONS else []
        return httpx.Response(200, json={"data": data, "meta": {"count": len(data)}})

    def _flights(self, request):
        if self.fail_flights_times > 0:
            self.fail_flights_times -= 1
            return httpx.Response(500, json={"errors": [{"status": 500}]})
        p = request.url.params
        o, d, day = p["originLocationCode"], p["destinationLocationCode"], p["departureDate"]
        offers = [
            _flight(0, o, d, day, self.flight_price[0], self.flight_stops),
            _flight(1, o, d, day, self.flight_price[1], self.flight_stops, "PT5H00M"),
        ]
        return httpx.Response(200, json={"data": offers, "meta": {"count": 2}})

    def _flights_post(self, request):
        if self.fail_flights_times > 0:
            return httpx.Response(500, json={"errors": [{"status": 500}]})
        return httpx.Response(200, json={"data": []})

    def _hotels(self, request):
        code = request.url.params["cityCode"]
        data = (
            []
            if self.no_hotels
            else [{"hotelId": h, "name": n, "chainCode": "XX", "iataCode": code} for h, n, _, _ in HOTELS]
        )
        return httpx.Response(200, json={"data": data})

    def _offers(self, request):
        p = request.url.params
        ids = p["hotelIds"].split(",")
        data = [
            {
                "hotel": {"hotelId": h, "name": n},
                "available": True,
                "offers": [
                    {
                        "id": f"o{h}",
                        "checkInDate": p["checkInDate"],
                        "checkOutDate": p["checkOutDate"],
                        "room": {"type": room},
                        "price": {"currency": "EUR", "total": str(price)},
                    }
                ],
            }
            for h, n, room, price in HOTELS
            if h in ids
        ]
        return httpx.Response(200, json={"data": data})

    def _forecast(self, request):
        start = request.url.params.get("start_date", "2027-01-01")
        from datetime import date, timedelta

        d0 = date.fromisoformat(start)
        days = [str(d0 + timedelta(days=i)) for i in range(7)]
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": days,
                    "temperature_2m_max": [20] * 7,
                    "temperature_2m_min": [10] * 7,
                    "precipitation_sum": [0, 8, 0, 0, 1, 0, 0],
                }
            },
        )

    def _insert(self, request):
        import json

        body = json.loads(request.content)
        self.inserted.append(body)
        return httpx.Response(
            200, json={"id": f"ev{len(self.inserted)}", "status": "confirmed", "htmlLink": "x", **body}
        )


@pytest.fixture
def travel_api():
    with respx.mock(assert_all_called=False) as router:
        yield MockTravelApi(router).install()
