"""Transport agent: searches and compares flights (outbound + return)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from vacation_agent.agents.base import AgentRun, react_loop
from vacation_agent.agents.protocol import AgentResult, AgentTask, specialist
from vacation_agent.tools.runner import ToolRequest
from vacation_agent.tools.travel import TRANSPORT_TOOLS

TOOLS = {t.name: t for t in TRANSPORT_TOOLS}

SYSTEM = """You are the transport specialist of a travel agency team.
Find flights with the search_flights tool:
- the outbound flight (origin -> destination on the departure date) AND the return flight
  (destination -> origin on the return date). Issue BOTH searches in the same step (parallel calls).
- If a search returns no offers, retry once with the day before or after.
Use only IATA codes from the task. When done, answer with one short sentence (no prices invented)."""


def _offers_for(run: AgentRun, origin: str, destination: str) -> list[dict[str, Any]]:
    offers: dict[str, dict[str, Any]] = {}
    for call in run.calls:
        if call.name != "search_flights" or not isinstance(call.result, dict) or not call.result.get("ok"):
            continue
        if (
            str(call.args.get("origin_iata", "")).upper() == origin
            and str(call.args.get("destination_iata", "")).upper() == destination
        ):
            for o in call.result.get("offers", []):
                offers[f"{o['flight_number']}|{o['departure_at']}"] = o
    return sorted(offers.values(), key=lambda o: o["total_price"])


def _errors(run: AgentRun) -> list[str]:
    return [str(c.result.get("error")) for c in run.calls if isinstance(c.result, dict) and c.result.get("ok") is False]


@specialist("TransportAgent")
async def run_transport(task: AgentTask) -> AgentResult:
    i = task.inputs
    origin, destination = i["origin_iata"].upper(), i["destination_iata"].upper()
    depart, ret = date.fromisoformat(i["departure_date"]), date.fromisoformat(i["return_date"])
    adults, children = int(i.get("adults", 1)), int(i.get("children", 0))

    run = await react_loop(
        "transport",
        SYSTEM,
        f"Task: {task.objective}\nInputs: origin={origin}, destination={destination}, "
        f"departure_date={depart}, return_date={ret}, adults={adults}, children={children}\n"
        f"Constraints: {'; '.join(task.constraints) or 'none'}",
        TOOLS,
    )

    # Guarantee: both directions searched, whatever the model did.
    def req(o: str, d: str, day: date) -> ToolRequest:
        return ToolRequest(
            "search_flights",
            {
                "origin_iata": o,
                "destination_iata": d,
                "departure_date": str(day),
                "adults": adults,
                "children": children,
            },
        )

    missing = []
    if not _offers_for(run, origin, destination):
        missing.append(req(origin, destination, depart))
    if not _offers_for(run, destination, origin):
        missing.append(req(destination, origin, ret))
    if missing:
        await run.run_requests("transport", missing, TOOLS, thought="Required flight searches (deterministic).")

    # Fallback: shift by one day if a direction still has no offers.
    shifted = []
    if not _offers_for(run, origin, destination):
        shifted.append(req(origin, destination, depart - timedelta(days=1)))
    if not _offers_for(run, destination, origin):
        shifted.append(req(destination, origin, ret + timedelta(days=1)))
    if shifted:
        await run.run_requests("transport", shifted, TOOLS, thought="No offers on the exact dates; trying ±1 day.")

    outbound, inbound = _offers_for(run, origin, destination), _offers_for(run, destination, origin)
    notes = []
    if shifted:
        notes.append("Pentru cel puțin un sens nu existau zboruri în data exactă; am căutat cu o zi diferență.")
    status = "ok" if outbound and inbound else "partial" if outbound or inbound else "failed"
    if status == "failed" and _errors(run):
        notes.append("Serviciul de zboruri a returnat erori: " + "; ".join(_errors(run)[:2]))
    return AgentResult(
        task_id=task.task_id,
        agent="transport",
        status=status,
        data={"outbound": outbound, "inbound": inbound, "summary": run.final_text},
        notes=notes,
        messages=run.messages,
    )
