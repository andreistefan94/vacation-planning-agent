"""Travel tools: deterministic wrappers over the mock (Amadeus / Open-Meteo / Google Calendar) APIs.

Every tool returns a JSON-serializable dict with ``ok`` plus either the normalized data or an
``error`` message. Transient failures are retried inside the client; what remains is reported to
the agent as a structured error instead of raising, so the agent can fall back or escalate.
"""

from __future__ import annotations

import asyncio
import math
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from langchain_core.tools import StructuredTool

from vacation_agent.config import get_settings
from vacation_agent.tools.client import ToolError, get_client
from vacation_agent.tools.schemas import (
    BudgetInput,
    BudgetResult,
    CalendarCheckInput,
    CalendarEventInput,
    DailyForecast,
    FlightOffer,
    HotelOffer,
    HotelOffersInput,
    HotelSummary,
    ListHotelsInput,
    Location,
    ResolveLocationInput,
    SearchFlightsInput,
    WeatherInput,
)

MAX_HOTEL_IDS = 8


def _err(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def _ascii(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def rooms_needed(adults: int) -> int:
    """One room per two adults; children share their parents' room."""
    return max(1, math.ceil(adults / 2))


# ---------------------------------------------------------------- locations


async def resolve_location(keyword: str) -> dict[str, Any]:
    """Resolve a city name or IATA code to its IATA city code and coordinates (exact match)."""
    client = get_client()
    candidates: list[str] = []
    for variant in (keyword.strip(), _ascii(keyword.strip()), _ascii(keyword.strip()).title()):
        if variant and variant not in candidates:
            candidates.append(variant)
    try:
        for variant in candidates:
            data = await client.request("GET", "/reference-data/locations", params={"keyword": variant})
            items = data.get("data", [])
            if items:
                matches = [
                    Location(
                        iata_code=i["iataCode"],
                        name=i.get("name", ""),
                        city_name=i.get("address", {}).get("cityName", ""),
                        city_code=i.get("address", {}).get("cityCode", i["iataCode"]),
                        country_code=i.get("address", {}).get("countryCode", ""),
                        latitude=i.get("geoCode", {}).get("latitude", 0.0),
                        longitude=i.get("geoCode", {}).get("longitude", 0.0),
                        sub_type=i.get("subType", ""),
                    ).model_dump()
                    for i in items
                ]
                # Prefer the CITY entry first.
                matches.sort(key=lambda m: m["sub_type"] != "CITY")
                return {"ok": True, "keyword": keyword, "matches": matches}
        return {
            "ok": True,
            "keyword": keyword,
            "matches": [],
            "note": "No exact match. Use the English city name (e.g. 'Rome', not 'Roma') or an IATA code.",
        }
    except ToolError as exc:
        return _err(exc)


# ---------------------------------------------------------------- flights


def _normalize_flight(raw: dict[str, Any], travellers: int) -> FlightOffer:
    segments = [s for it in raw.get("itineraries", []) for s in it.get("segments", [])]
    first, last = segments[0], segments[-1]
    # Connections between segments count as stops too.
    stops = sum(int(s.get("numberOfStops", 0)) for s in segments) + max(0, len(segments) - 1)
    price = raw.get("price", {})
    grand_total = float(price.get("grandTotal") or price.get("total") or 0)
    priced_travellers = max(1, len(raw.get("travelerPricings") or [1]))
    per_person = grand_total / priced_travellers
    return FlightOffer(
        offer_id=str(raw.get("id", "")),
        carrier=first.get("carrierCode", ""),
        flight_number=f"{first.get('carrierCode', '')}{first.get('number', '')}",
        origin=first.get("departure", {}).get("iataCode", ""),
        destination=last.get("arrival", {}).get("iataCode", ""),
        departure_at=first.get("departure", {}).get("at", ""),
        arrival_at=last.get("arrival", {}).get("at", ""),
        duration=(raw.get("itineraries") or [{}])[0].get("duration", ""),
        stops=stops,
        direct=stops == 0,
        price_per_person=round(per_person, 2),
        travellers=travellers,
        total_price=round(per_person * travellers, 2),
        currency=price.get("currency", "EUR"),
        seats_available=raw.get("numberOfBookableSeats"),
    )


async def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: date,
    adults: int = 1,
    children: int = 0,
) -> dict[str, Any]:
    """Search one-way flight offers. Prices are converted to totals for all travellers."""
    client = get_client()
    travellers = adults + children
    params = {
        "originLocationCode": origin_iata.upper(),
        "destinationLocationCode": destination_iata.upper(),
        "departureDate": str(departure_date),
        "adults": travellers,
    }
    try:
        try:
            data = await client.request("GET", "/shopping/flight-offers", params=params)
        except ToolError:
            # Fallback: the same search through the JSON-body endpoint.
            body = {
                "originDestinations": [
                    {
                        "id": "1",
                        "originLocationCode": params["originLocationCode"],
                        "destinationLocationCode": params["destinationLocationCode"],
                        "departureDateTimeRange": {"date": params["departureDate"]},
                    }
                ],
                "travelers": [{"id": str(i + 1), "travelerType": "ADULT"} for i in range(travellers)],
            }
            data = await client.request("POST", "/shopping/flight-offers", json=body)
        offers = []
        for raw in data.get("data", []):
            try:
                offers.append(_normalize_flight(raw, travellers))
            except (KeyError, IndexError, ValueError, TypeError):
                continue
        offers.sort(key=lambda o: o.total_price)
        return {
            "ok": True,
            "origin": params["originLocationCode"],
            "destination": params["destinationLocationCode"],
            "date": params["departureDate"],
            "travellers": travellers,
            "offers": [o.model_dump() for o in offers],
        }
    except ToolError as exc:
        return _err(exc)


# ---------------------------------------------------------------- hotels


async def list_hotels_in_city(city_code: str) -> dict[str, Any]:
    """List hotels in a city (by IATA city code). Use get_hotel_offers afterwards for prices."""
    try:
        data = await get_client().request(
            "GET", "/reference-data/locations/hotels/by-city", params={"cityCode": city_code.upper()}
        )
        hotels = [
            HotelSummary(hotel_id=h["hotelId"], name=h.get("name", ""), chain_code=h.get("chainCode", ""))
            for h in data.get("data", [])
        ]
        return {"ok": True, "city_code": city_code.upper(), "hotels": [h.model_dump() for h in hotels]}
    except ToolError as exc:
        return _err(exc)


async def get_hotel_offers(
    hotel_ids: list[str],
    check_in_date: date,
    check_out_date: date,
    adults: int = 1,
    children: int = 0,
) -> dict[str, Any]:
    """Price and availability for given hotels. Prices are per night per room in the API; we return totals."""
    nights = (check_out_date - check_in_date).days
    if nights <= 0:
        return {"ok": False, "error": "check_out_date must be after check_in_date"}
    rooms = rooms_needed(adults)
    try:
        data = await get_client().request(
            "GET",
            "/shopping/hotel-offers",
            params={
                "hotelIds": ",".join(hotel_ids[:MAX_HOTEL_IDS]),
                "checkInDate": str(check_in_date),
                "checkOutDate": str(check_out_date),
                "adults": adults,
            },
        )
        offers: list[HotelOffer] = []
        for item in data.get("data", []):
            hotel = item.get("hotel", {})
            for raw in item.get("offers", [])[:1]:
                per_night = float(raw.get("price", {}).get("total") or 0)
                offers.append(
                    HotelOffer(
                        hotel_id=hotel.get("hotelId", ""),
                        name=hotel.get("name", ""),
                        available=bool(item.get("available", False)),
                        room_type=raw.get("room", {}).get("type", ""),
                        price_per_night=per_night,
                        nights=nights,
                        rooms=rooms,
                        total_price=round(per_night * nights * rooms, 2),
                        currency=raw.get("price", {}).get("currency", "EUR"),
                        offer_id=raw.get("id", ""),
                    )
                )
        offers.sort(key=lambda o: o.total_price)
        return {
            "ok": True,
            "check_in": str(check_in_date),
            "check_out": str(check_out_date),
            "nights": nights,
            "rooms": rooms,
            "guests": adults + children,
            "offers": [o.model_dump() for o in offers],
        }
    except ToolError as exc:
        return _err(exc)


# ---------------------------------------------------------------- weather


async def _open_meteo(latitude: float, longitude: float, start: date) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": str(start),
        "end_date": str(start + timedelta(days=6)),
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
        return r.json()


async def get_weather_forecast(latitude: float, longitude: float, start_date: date | None = None) -> dict[str, Any]:
    """7-day daily weather forecast (max/min temperature in °C, precipitation in mm)."""
    start = start_date or date.today()
    data: dict[str, Any] | None = None
    source = "mock"
    if get_settings().tools_backend == "real":
        try:
            data = await _open_meteo(latitude, longitude, start)
            source = "open-meteo"
        except Exception:  # noqa: BLE001 - real API is optional, fall back to the mock
            data = None
    try:
        if data is None:
            data = await get_client().request(
                "GET",
                "/v1/forecast",
                params={"latitude": latitude, "longitude": longitude, "start_date": str(start)},
            )
        daily = data.get("daily", {})
        days = [
            DailyForecast(date=d, t_max=tx, t_min=tn, precipitation_mm=p).model_dump()
            for d, tx, tn, p in zip(
                daily.get("time", []),
                daily.get("temperature_2m_max", []),
                daily.get("temperature_2m_min", []),
                daily.get("precipitation_sum", []),
                strict=False,
            )
        ]
        return {"ok": True, "source": source, "days": days}
    except ToolError as exc:
        return _err(exc)


# ---------------------------------------------------------------- calendar


def _event_day_span(event: dict[str, Any]) -> tuple[date, date] | None:
    start, end = event.get("start", {}), event.get("end", {})
    try:
        if "date" in start:
            s = date.fromisoformat(start["date"])
            e = date.fromisoformat(end.get("date", start["date"])) - timedelta(days=1)
        else:
            s = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00")).date()
            e = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00")).date()
        return s, max(s, e)
    except (KeyError, ValueError):
        return None


async def check_calendar_availability(start_date: date, end_date: date) -> dict[str, Any]:
    """Return the user's busy days between start_date and end_date (inclusive)."""
    settings = get_settings()
    try:
        data = await get_client().request("GET", f"/calendar/v3/calendars/{settings.calendar_id}/events", auth=True)
    except ToolError as exc:
        return _err(exc)
    busy: dict[str, str] = {}
    for ev in data.get("items", []):
        span = _event_day_span(ev)
        if not span:
            continue
        d = max(span[0], start_date)
        while d <= min(span[1], end_date):
            busy[str(d)] = ev.get("summary", "busy")
            d += timedelta(days=1)
    return {
        "ok": True,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "free": not busy,
        "busy_days": [{"date": k, "summary": v} for k, v in sorted(busy.items())],
    }


async def add_calendar_event(summary: str, start_date: date, end_date: date, description: str = "") -> dict[str, Any]:
    """Insert the trip into the user's calendar. EXTERNAL ACTION: only call after explicit user approval."""
    settings = get_settings()
    body = {
        "summary": summary,
        "description": description,
        "start": {"date": str(start_date)},
        "end": {"date": str(end_date + timedelta(days=1))},  # Google all-day end is exclusive
    }
    try:
        data = await get_client().request(
            "POST", f"/calendar/v3/calendars/{settings.calendar_id}/events", auth=True, json=body
        )
        return {"ok": True, "event_id": data.get("id"), "status": data.get("status"), "link": data.get("htmlLink")}
    except ToolError as exc:
        return _err(exc)


# ---------------------------------------------------------------- budget


async def calculate_budget(
    flight_total: float,
    hotel_total: float,
    budget: float | None = None,
    extras: float = 0,
    currency: str = "EUR",
) -> dict[str, Any]:
    """Deterministic budget arithmetic: total cost, remaining amount and whether it fits the budget."""
    total = round(flight_total + hotel_total + extras, 2)
    result = BudgetResult(
        flight_total=round(flight_total, 2),
        hotel_total=round(hotel_total, 2),
        extras=round(extras, 2),
        total=total,
        budget=budget,
        remaining=None if budget is None else round(budget - total, 2),
        within_budget=None if budget is None else total <= budget,
        currency=currency,
    )
    return {"ok": True, **result.model_dump()}


# ---------------------------------------------------------------- registry


def _tool(fn, schema, name: str | None = None) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=fn, name=name or fn.__name__, description=(fn.__doc__ or "").strip(), args_schema=schema
    )


resolve_location_tool = _tool(resolve_location, ResolveLocationInput)
search_flights_tool = _tool(search_flights, SearchFlightsInput)
list_hotels_tool = _tool(list_hotels_in_city, ListHotelsInput)
hotel_offers_tool = _tool(get_hotel_offers, HotelOffersInput)
weather_tool = _tool(get_weather_forecast, WeatherInput)
calendar_check_tool = _tool(check_calendar_availability, CalendarCheckInput)
calendar_add_tool = _tool(add_calendar_event, CalendarEventInput)
budget_tool = _tool(calculate_budget, BudgetInput)

TRANSPORT_TOOLS = [search_flights_tool]
ACCOMMODATION_TOOLS = [list_hotels_tool, hotel_offers_tool]
ALL_TOOLS = {
    t.name: t
    for t in [
        resolve_location_tool,
        search_flights_tool,
        list_hotels_tool,
        hotel_offers_tool,
        weather_tool,
        calendar_check_tool,
        calendar_add_tool,
        budget_tool,
    ]
}


async def gather_dict(**coros) -> dict[str, Any]:
    """Run coroutines in parallel and return results by key."""
    keys = list(coros)
    results = await asyncio.gather(*coros.values())
    return dict(zip(keys, results, strict=True))
