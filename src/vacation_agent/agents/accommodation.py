"""Accommodation agent: finds hotels, checks prices/availability and the user's preferences."""

from __future__ import annotations

from typing import Any

from vacation_agent.agents.base import AgentRun, react_loop
from vacation_agent.agents.protocol import AgentResult, AgentTask, specialist
from vacation_agent.tools.runner import ToolRequest
from vacation_agent.tools.travel import ACCOMMODATION_TOOLS, MAX_HOTEL_IDS

TOOLS = {t.name: t for t in ACCOMMODATION_TOOLS}

SYSTEM = f"""You are the accommodation specialist of a travel agency team.
Steps (sequential): 1) list_hotels_in_city for the city code; 2) get_hotel_offers for up to
{MAX_HOTEL_IDS} of the returned hotel ids, with the exact check-in/check-out dates and guests.
Never make up hotel ids. When done, answer with one short sentence (no invented prices)."""

ROOM_RANK = {"STANDARD": 0, "SUPERIOR": 1, "DELUXE": 2, "SUITE": 3}


def _collect(run: AgentRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hotels: dict[str, dict] = {}
    offers: dict[str, dict] = {}
    for call in run.calls:
        res = call.result if isinstance(call.result, dict) else {}
        if not res.get("ok"):
            continue
        if call.name == "list_hotels_in_city":
            for h in res.get("hotels", []):
                hotels[h["hotel_id"]] = h
        elif call.name == "get_hotel_offers":
            for o in res.get("offers", []):
                offers[o["hotel_id"]] = o
    return list(hotels.values()), list(offers.values())


@specialist("AccommodationAgent")
async def run_accommodation(task: AgentTask) -> AgentResult:
    i = task.inputs
    city, check_in, check_out = i["city_code"].upper(), i["check_in"], i["check_out"]
    adults, children = int(i.get("adults", 1)), int(i.get("children", 0))

    run = await react_loop(
        "accommodation",
        SYSTEM,
        f"Task: {task.objective}\nInputs: city_code={city}, check_in_date={check_in}, "
        f"check_out_date={check_out}, adults={adults}, children={children}\n"
        f"User preferences to respect: {'; '.join(task.constraints) or 'none'}",
        TOOLS,
        max_steps=4,
    )

    hotels, offers = _collect(run)
    if not hotels:
        await run.run_requests(
            "accommodation",
            [ToolRequest("list_hotels_in_city", {"city_code": city})],
            TOOLS,
            thought="Required hotel listing (deterministic).",
        )
        hotels, offers = _collect(run)
    if hotels and not offers:
        await run.run_requests(
            "accommodation",
            [
                ToolRequest(
                    "get_hotel_offers",
                    {
                        "hotel_ids": [h["hotel_id"] for h in hotels[:MAX_HOTEL_IDS]],
                        "check_in_date": check_in,
                        "check_out_date": check_out,
                        "adults": adults,
                        "children": children,
                    },
                )
            ],
            TOOLS,
            thought="Required price/availability check (deterministic).",
        )
        hotels, offers = _collect(run)

    # Availability must be verified before an option can be used in a variant.
    available = sorted((o for o in offers if o.get("available")), key=lambda o: o["total_price"])
    notes = []
    if offers and not available:
        notes.append("Niciunul dintre hotelurile verificate nu are disponibilitate pe aceste date.")
    if any("breakfast" in c.lower() or "mic dejun" in c.lower() for c in task.constraints):
        notes.append("Serviciul de cazare nu indică dacă micul dejun este inclus; trebuie verificat la hotel.")
    for o in available:
        o["room_rank"] = ROOM_RANK.get(o.get("room_type", ""), 0)
    status = "ok" if available else "failed"
    return AgentResult(
        task_id=task.task_id,
        agent="accommodation",
        status=status,
        data={"offers": available, "hotels_checked": len(offers), "summary": run.final_text},
        notes=notes,
        messages=run.messages,
    )
