from datetime import date

from vacation_agent.tools import travel
from vacation_agent.tools.runner import ToolRequest, execute_tools


async def test_resolve_location_folds_diacritics(travel_api):
    res = await travel.resolve_location("Chișinău")
    assert res["ok"] and res["matches"][0]["city_code"] == "KIV"


async def test_resolve_location_no_match_is_not_an_error(travel_api):
    res = await travel.resolve_location("undeva cald")
    assert res["ok"] and res["matches"] == [] and "English" in res["note"]


async def test_flight_prices_are_multiplied_by_travellers(travel_api):
    res = await travel.search_flights("KIV", "ROM", date(2026, 12, 18), adults=2, children=1)
    cheapest = res["offers"][0]
    assert cheapest["price_per_person"] == 120.0
    assert cheapest["total_price"] == 360.0 and cheapest["direct"]


async def test_flights_retry_after_server_error(travel_api):
    travel_api.fail_flights_times = 1  # first attempt 500, retry succeeds
    res = await travel.search_flights("KIV", "ROM", date(2026, 12, 18))
    assert res["ok"] and res["offers"]


async def test_flights_error_reported_after_retries_and_fallback(travel_api):
    travel_api.fail_flights_times = 100
    res = await travel.search_flights("KIV", "ROM", date(2026, 12, 18))
    assert res["ok"] is False and "error" in res


async def test_hotel_offers_totals_per_night_and_rooms(travel_api):
    res = await travel.get_hotel_offers(["H1", "H2"], date(2026, 12, 18), date(2026, 12, 25), adults=3)
    h1 = next(o for o in res["offers"] if o["hotel_id"] == "H1")
    assert h1["nights"] == 7 and h1["rooms"] == 2 and h1["total_price"] == 700.0


async def test_calendar_busy_days_and_insert(travel_api):
    travel_api.calendar_events.append(
        {"summary": "Work", "start": {"date": "2026-12-20"}, "end": {"date": "2026-12-22"}}
    )
    res = await travel.check_calendar_availability(date(2026, 12, 18), date(2026, 12, 25))
    assert [d["date"] for d in res["busy_days"]] == ["2026-12-20", "2026-12-21"]
    ins = await travel.add_calendar_event("Trip", date(2026, 12, 18), date(2026, 12, 25))
    assert ins["ok"] and travel_api.inserted[0]["end"] == {"date": "2026-12-26"}


async def test_budget_tool():
    res = await travel.calculate_budget(flight_total=600, hotel_total=900, budget=1400)
    assert res["total"] == 1500 and res["within_budget"] is False and res["remaining"] == -100


async def test_runner_records_parallel_calls_and_agent_level_retry(travel_api):
    travel_api.fail_flights_times = 100
    batch = await execute_tools(
        [
            ToolRequest(
                "search_flights", {"origin_iata": "KIV", "destination_iata": "ROM", "departure_date": "2026-12-18"}
            ),
            ToolRequest("calculate_budget", {"flight_total": 1, "hotel_total": 2}),
        ],
        travel.ALL_TOOLS,
        agent="test",
    )
    first = batch.messages[0]
    assert len(first.tool_calls) == 2  # parallel calls in one assistant message
    retry = [m for m in batch.messages if m.type == "ai"][1]  # agent-level retry, visible in the history
    assert [tc["name"] for tc in retry.tool_calls] == ["search_flights"]
    tool_ids = {m.tool_call_id for m in batch.messages if m.type == "tool"}
    call_ids = {tc["id"] for m in batch.messages if m.type == "ai" for tc in m.tool_calls}
    assert tool_ids == call_ids
