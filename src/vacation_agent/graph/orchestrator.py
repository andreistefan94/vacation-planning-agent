"""Orchestrator graph (LangGraph).

Flow of one planning:

    start_turn ─▶ check_requirements ─▶ resolve_places ─▶ delegate ─▶ build_variants ─▶ present_variants
        │               │ (missing)         │ (several cities)            │ (no direct)       │
        ▼               ▼                   ▼                             ▼ (over budget)     ▼
     respond         respond      present_destinations ⇢ choose_destination   present_change ⇢ approve_change
                                                                          present_overrun ⇢ approve_overrun
    choose_variant ─▶ generate_itinerary ─▶ present_final ⇢ final_approval ─▶ finalize ─▶ respond

``⇢`` marks a human-in-the-loop interrupt: the run is suspended with ``interrupt()`` and resumed with
``Command(resume=<user answer>)`` on the same thread (thread_id = conversation_id), continuing from
that node instead of restarting. Missing information is collected with ordinary clarification
questions (the run ends normally and the next message continues from the stored state).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from vacation_agent.agents.accommodation import run_accommodation
from vacation_agent.agents.activities import run_activities
from vacation_agent.agents.conversational import (
    Decision,
    analyze_turn,
    answer_from_guides,
    interpret_decision,
    write_clarification,
)
from vacation_agent.agents.protocol import AgentTask
from vacation_agent.agents.transport import run_transport
from vacation_agent.graph import planning as P
from vacation_agent.graph.requirements import (
    describe_request,
    merge_request,
    missing_slots,
    pick_free_window,
    resolve_dates,
    search_window,
    travellers,
)
from vacation_agent.graph.state import TripState
from vacation_agent.memory.preferences import MEMORY_TOOLS, prefers_direct_flights
from vacation_agent.memory.store import get_store
from vacation_agent.rag.retriever import RAG_TOOLS, format_citation
from vacation_agent.tools.runner import ToolBatchResult, ToolRequest, execute_tools
from vacation_agent.tools.travel import ALL_TOOLS

log = logging.getLogger(__name__)

TOOLS = {**ALL_TOOLS, **MEMORY_TOOLS, **RAG_TOOLS}

HONESTY_NOTE = (
    "Precizare importantă: nu pot cumpăra bilete de avion, nu pot rezerva camere de hotel și nu pot efectua "
    "plăți — nu am acces la astfel de sisteme. Vă pot construi un itinerariu complet, cu prețuri reale din "
    "căutare, iar după aprobarea dvs. îl pot adăuga în calendar; rezervările le faceți separat."
)
CORE_TRIP_FIELDS = (
    "destination",
    "destination_city_en",
    "origin_city_en",
    "start_date",
    "end_date",
    "month",
    "duration_days",
    "adults",
    "children",
    "budget",
)


def today() -> date:
    return date.today()


async def run_tools(requests: list[ToolRequest], thought: str = "") -> ToolBatchResult:
    return await execute_tools(requests, TOOLS, agent="orchestrator", thought=thought)


def _last_human(state: TripState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


async def decide(answer: str, payload: dict[str, Any]) -> Decision:
    """Interpret the user's answer to a HITL request.

    A click on an option button (the UI sends the option text verbatim) is matched in code;
    only free-text answers go to the conversational agent's LLM.
    """
    options = payload.get("options") or []
    normalized = answer.strip().casefold()
    for i, option in enumerate(options):
        if normalized == option.strip().casefold():
            return Decision(action="select", option_index=i)
    return await interpret_decision(answer, payload["kind"], payload["question"], options, today())


def _location(match: dict[str, Any], query: str) -> dict[str, Any]:
    return {
        "query": query,
        "iata": match["city_code"] or match["iata_code"],
        "city_name": match["city_name"] or match["name"].title(),
        "country_code": match["country_code"],
        "latitude": match["latitude"],
        "longitude": match["longitude"],
    }


SEARCH_RESET = {
    "transport": None,
    "accommodation": None,
    "activities": None,
    "search_signature": None,
    "variants": [],
    "selected_variant": None,
    "overrun_approved": False,
    "change_approved": False,
    "itinerary": None,
}


# =============================================================== turn entry


async def start_turn(state: TripState) -> Command[Literal["check_requirements", "respond"]]:
    user_id = state["user_id"]
    message = _last_human(state)
    messages: list = []
    parts: list[str] = []
    prefs = list(state.get("preferences") or [])
    update: dict[str, Any] = {}

    if not state.get("memory_loaded"):
        batch = await run_tools(
            [ToolRequest("get_user_preferences", {"user_id": user_id})], "Loading long-term preferences."
        )
        messages += batch.messages
        result = next(iter(batch.results.values()))
        prefs = result.get("preferences", []) if result.get("ok") else []
        update["memory_loaded"] = True

    stage = state.get("stage") or "collecting"
    req = state.get("request") or {}
    analysis = await analyze_turn(
        message, state.get("messages", [])[:-1], req, state.get("pending_preference"), stage, today()
    )
    intents = set(analysis.intents)
    trip_update = analysis.trip.model_dump()
    new_req = merge_request(req, trip_update)
    trip_mentioned = any(trip_update.get(k) not in (None, [], "") for k in CORE_TRIP_FIELDS)

    # ---- long-term memory: confirmation of a pending preference
    pending = state.get("pending_preference")
    if pending and analysis.preference_confirmation == "yes":
        batch = await run_tools(
            [
                ToolRequest(
                    "save_user_preference",
                    {"user_id": user_id, "text": pending["text"], "category": pending.get("category", "general")},
                )
            ],
            "The user explicitly confirmed: persisting the preference.",
        )
        messages += batch.messages
        prefs = get_store().get_preferences(user_id)
        parts.append(
            f"Am reținut preferința „{pending['text']}” pentru călătoriile viitoare. O puteți vedea sau șterge oricând."
        )
        pending = None
    elif pending and analysis.preference_confirmation == "no":
        parts.append("Am înțeles, nu rețin această preferință permanent.")
        pending = None

    # ---- new preference candidates: permanent ones need explicit confirmation
    known = {p["text"].lower() for p in prefs}
    for cand in analysis.preference_candidates:
        if cand.category == "direct_flights" and new_req.get("direct_flights_only") is None:
            new_req["direct_flights_only"] = True  # applies to the current trip anyway
        if cand.explicitly_permanent and cand.text.lower() not in known and pending is None:
            pending = cand.model_dump()
            parts.append(
                f"Doriți să rețin „{cand.text}” ca preferință permanentă, pentru călătoriile viitoare? (da/nu)"
            )

    # ---- conflict between a stored preference and the current request
    if new_req.get("direct_flights_only") is False and prefers_direct_flights(prefs):
        parts.append(
            "Observ că de obicei preferați zboruri directe; pentru această călătorie țin cont de cererea "
            "actuală și accept și zboruri cu escală."
        )

    # ---- user control over memory
    if "forget_preferences" in intents:
        get_store().clear_preferences(user_id)
        prefs, pending = [], None
        parts.append("Am șters toate preferințele pe care le aveam reținute despre dvs.")
    elif "show_preferences" in intents:
        texts = [p["text"] for p in prefs]
        parts.append(
            "Preferințele reținute: " + "; ".join(texts) + "."
            if texts
            else "Nu am nicio preferință reținută despre dvs."
        )

    # ---- honesty about bookings
    if "booking_request" in intents:
        parts.append(HONESTY_NOTE)

    # ---- descriptive questions -> RAG
    if "guide_question" in intents and analysis.guide_question_en:
        city = analysis.guide_city_en or new_req.get("destination_city_en")
        batch = await run_tools(
            [ToolRequest("search_travel_guide", {"query": analysis.guide_question_en, "city": city})],
            "Descriptive question: searching the travel guides.",
        )
        messages += batch.messages
        res = next(iter(batch.results.values()))
        results = res.get("results", []) if res.get("ok") else []
        if results:
            answer = await answer_from_guides(message, results, analysis.language)
            sources = "\n".join(f"[{i + 1}] {format_citation(r)}" for i, r in enumerate(results))
            parts.append(f"{answer}\n\nSurse:\n{sources}")
        else:
            parts.append(
                (
                    "Ghidurile turistice pe care le am la dispoziție (Wikivoyage, 10 destinații) nu conțin "
                    "informații despre această întrebare, așa că nu vă pot oferi un răspuns fără să inventez. "
                    + (res.get("note") or "")
                ).strip()
            )

    planning = "plan_trip" in intents or "booking_request" in intents or trip_mentioned
    if stage == "done" and trip_mentioned:
        update.update(SEARCH_RESET)
        update["calendar_event"] = None
        update["rejected_variants"] = []
        stage = "collecting"

    update.update(
        {
            "messages": messages,
            "request": new_req,
            "preferences": prefs,
            "pending_preference": pending,
            "stage": stage,
            "reply_parts": parts,
            "turn": {"intents": sorted(intents), "planning": planning, "language": analysis.language},
        }
    )
    has_trip = any(new_req.get(k) for k in ("destination", "destination_city_en"))
    go = "check_requirements" if planning and (has_trip or "plan_trip" in intents) and stage != "done" else "respond"
    return Command(update=update, goto=go)


# =============================================================== requirements & places


async def check_requirements(state: TripState) -> Command[Literal["resolve_places", "respond"]]:
    req = state.get("request") or {}
    missing = missing_slots(req, today())
    if missing:
        text = await write_clarification(
            describe_request(req), missing, (state.get("turn") or {}).get("language", "ro")
        )
        return Command(
            update={
                "reply_parts": (state.get("reply_parts") or []) + [text],
                "stage": "collecting",
            },
            goto="respond",
        )
    return Command(goto="resolve_places")


async def resolve_places(
    state: TripState,
) -> Command[Literal["delegate", "present_destinations", "respond"]]:
    req = dict(state.get("request") or {})
    parts = list(state.get("reply_parts") or [])
    requests: list[ToolRequest] = []
    origin = state.get("origin")
    if not origin or origin.get("query", "").lower() != (req.get("origin_city_en") or "").lower():
        origin = None
        requests.append(ToolRequest("resolve_location", {"keyword": req["origin_city_en"]}, id="call_origin"))

    destination = state.get("destination")
    city_query = req.get("destination_city_en") if req.get("destination_kind") in (None, "city") else None
    if destination and city_query and destination.get("query", "").lower() == city_query.lower():
        pass
    else:
        destination = None
        if city_query:
            requests.append(ToolRequest("resolve_location", {"keyword": city_query}, id="call_dest"))
        else:
            for i, c in enumerate(req.get("candidate_cities_en", [])[:4]):
                requests.append(ToolRequest("resolve_location", {"keyword": c}, id=f"call_cand{i}"))

    window = None
    if not resolve_dates(req, today()):
        window = search_window(req, today())
        if window:
            requests.append(
                ToolRequest(
                    "check_calendar_availability",
                    {"start_date": str(window[0]), "end_date": str(window[1])},
                    id="call_calendar",
                )
            )

    batch = (
        await run_tools(requests, "Resolving locations and checking the calendar in parallel.") if requests else None
    )
    results = batch.results if batch else {}
    update: dict[str, Any] = {
        "messages": batch.messages if batch else [],
    }

    def first_match(call_id: str) -> dict[str, Any] | None:
        res = results.get(call_id) or {}
        matches = res.get("matches") or []
        return matches[0] if res.get("ok") and matches else None

    if "call_origin" in results:
        m = first_match("call_origin")
        if not m:
            parts.append(
                f"Nu am găsit orașul de plecare „{req['origin_city_en']}” în sistemul de zboruri. "
                "Puteți indica un alt oraș sau codul IATA al aeroportului?"
            )
            req["origin_city_en"] = None
            return Command(
                update={**update, "request": req, "reply_parts": parts, "stage": "collecting"}, goto="respond"
            )
        origin = _location(m, req["origin_city_en"])

    options: list[dict[str, Any]] = []
    if "call_dest" in results:
        m = first_match("call_dest")
        if not m:
            parts.append(
                f"Nu am găsit destinația „{city_query}” în sistemul de zboruri și cazare. "
                "Puteți indica un oraș apropiat (sau codul IATA)?"
            )
            req["destination_city_en"] = None
            req["destination"] = None
            return Command(
                update={**update, "request": req, "reply_parts": parts, "stage": "collecting"}, goto="respond"
            )
        destination = _location(m, city_query)
    elif not destination:
        seen = set()
        for cid in results:
            if cid.startswith("call_cand"):
                m = first_match(cid)
                if m and m["city_code"] not in seen:
                    seen.add(m["city_code"])
                    options.append(_location(m, m["city_name"]))
        if not options:
            parts.append(
                f"Pentru „{req.get('destination')}” nu am găsit orașe disponibile în sistemul de rezervări. "
                "Îmi puteți spune un oraș anume?"
            )
            return Command(update={**update, "reply_parts": parts, "stage": "collecting"}, goto="respond")
        if len(options) == 1:
            destination = options[0]

    if window:
        cal = results.get("call_calendar") or {}
        busy = {d["date"] for d in cal.get("busy_days", [])} if cal.get("ok") else set()
        start, end = pick_free_window(window, int(req["duration_days"]), busy)
        req["start_date"], req["end_date"] = str(start), str(end)
        note = "(zile libere în calendarul dvs.)" if cal.get("ok") else "(nu am putut verifica calendarul)"
        parts.append(
            f"Nu ați indicat datele exacte, așa că am presupus perioada {start:%d.%m.%Y} – {end:%d.%m.%Y} "
            f"{note}. Le puteți schimba oricând."
        )
        update["dates_assumed"] = True

    update.update({"request": req, "origin": origin, "reply_parts": parts})
    if destination is None:
        update["destination_options"] = options
        return Command(update={**update, "destination": None}, goto="present_destinations")
    if destination.get("query", "").lower() != (req.get("destination_city_en") or "").lower():
        req["destination_city_en"] = destination["city_name"]
        req["destination_kind"] = "city"
        destination["query"] = destination["city_name"]
    return Command(update={**update, "destination": destination, "request": req}, goto="delegate")


async def present_destinations(state: TripState) -> dict[str, Any]:
    options = state.get("destination_options") or []
    req = state.get("request") or {}
    labels = [f"{o['city_name']} ({o['country_code']})" for o in options]
    question = f"Pentru „{req.get('destination')}” se potrivesc mai multe orașe. Pe care îl alegeți?"
    text = "\n\n".join(
        (state.get("reply_parts") or [])
        + [question + "\n" + "\n".join(f"{i + 1}. {label}" for i, label in enumerate(labels))]
    )
    payload = {"kind": "choose_destination", "question": question, "options": labels, "reply": text}
    return {
        "messages": [AIMessage(content=text)],
        "pending_payload": payload,
        "stage": "choosing_destination",
        "reply_parts": [],
    }


async def choose_destination(
    state: TripState,
) -> Command[Literal["check_requirements", "present_destinations", "respond"]]:
    payload = state["pending_payload"]
    answer = interrupt(payload)
    decision = await decide(answer, payload)
    req = dict(state.get("request") or {})
    if decision.trip_changes:
        req = merge_request(req, decision.trip_changes.model_dump())
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=answer)],
        "reply_parts": [],
    }
    options = state.get("destination_options") or []
    if decision.action == "select" and decision.option_index is not None:
        chosen = dict(options[decision.option_index])
        req["destination_city_en"] = chosen["city_name"]
        req["destination_kind"] = "city"
        chosen["query"] = chosen["city_name"]
        return Command(
            update={**update, "request": req, "destination": chosen, **SEARCH_RESET}, goto="check_requirements"
        )
    if decision.action == "modify":
        return Command(update={**update, "request": req, **SEARCH_RESET}, goto="check_requirements")
    if decision.action == "reject":
        req["destination"] = req["destination_city_en"] = None
        req["candidate_cities_en"] = []
        return Command(
            update={
                **update,
                "request": req,
                "stage": "collecting",
                "reply_parts": ["Ce altă destinație ați prefera?"],
            },
            goto="respond",
        )
    return Command(
        update={**update, "reply_parts": ["Nu am înțeles alegerea. Vă rog indicați unul dintre orașele de mai jos."]},
        goto="present_destinations",
    )


# =============================================================== delegation to specialists


async def delegate(state: TripState) -> Command[Literal["build_variants", "respond"]]:
    req = state["request"]
    origin, dest = state["origin"], state["destination"]
    start, end = resolve_dates(req, today())
    adults, children = travellers(req)
    sig = json.dumps([origin["iata"], dest["iata"], str(start), str(end), adults, children])
    if sig == state.get("search_signature") and state.get("transport") and state.get("accommodation"):
        return Command(goto="build_variants")  # same search as before: reuse the results

    prefs = [p["text"] for p in state.get("preferences") or []]
    constraints = prefs + (req.get("notes") or [])
    if req.get("direct_flights_only") or prefers_direct_flights(state.get("preferences") or []):
        constraints.append("prefer direct flights")
    tasks = [
        AgentTask(
            agent="transport",
            objective="Find outbound and return flight offers.",
            inputs={
                "origin_iata": origin["iata"],
                "destination_iata": dest["iata"],
                "departure_date": str(start),
                "return_date": str(end),
                "adults": adults,
                "children": children,
            },
            constraints=constraints,
        ),
        AgentTask(
            agent="accommodation",
            objective="Find available hotels with prices for the whole stay.",
            inputs={
                "city_code": dest["iata"],
                "check_in": str(start),
                "check_out": str(end),
                "adults": adults,
                "children": children,
            },
            constraints=constraints,
        ),
        AgentTask(
            agent="activities",
            objective="Collect the weather forecast and guide knowledge to plan the days.",
            inputs={
                "city_name": dest["city_name"],
                "latitude": dest["latitude"],
                "longitude": dest["longitude"],
                "start_date": str(start),
                "days": (end - start).days + 1,
                "children": children,
            },
            constraints=constraints,
        ),
    ]
    transport, accommodation, activities = await asyncio.gather(
        run_transport(tasks[0]), run_accommodation(tasks[1]), run_activities(tasks[2])
    )
    results = (transport, accommodation, activities)
    messages = [m for r in results for m in r.messages]
    update: dict[str, Any] = {
        **SEARCH_RESET,
        "messages": messages,
        "transport": {"status": transport.status, **transport.data, "notes": transport.notes},
        "accommodation": {"status": accommodation.status, **accommodation.data, "notes": accommodation.notes},
        "activities": {"status": activities.status, **activities.data, "notes": activities.notes},
        "search_signature": sig,
        "stage": "searching",
    }
    problems = []
    if transport.status != "ok":
        problems.append(
            f"nu am găsit zboruri disponibile {origin['city_name']} ⇄ {dest['city_name']} "
            f"pentru {start:%d.%m} – {end:%d.%m}"
        )
    if accommodation.status != "ok":
        problems.append(f"nu am găsit cazare disponibilă în {dest['city_name']} pentru aceste date")
    if problems:
        parts = list(state.get("reply_parts") or [])
        parts.append(
            "Din păcate "
            + " și ".join(problems)
            + ". "
            + " ".join(transport.notes + accommodation.notes)
            + " Doriți să încercăm alte date sau o altă destinație?"
        )
        update["search_signature"] = None
        return Command(update={**update, "reply_parts": parts, "stage": "collecting"}, goto="respond")
    return Command(update=update, goto="build_variants")


# =============================================================== variants & budget


async def build_variants(
    state: TripState,
) -> Command[Literal["present_change", "present_overrun", "present_variants", "respond"]]:
    req = state["request"]
    prefs = state.get("preferences") or []
    prefer_direct = req.get("direct_flights_only")
    if prefer_direct is None:
        prefer_direct = prefers_direct_flights(prefs)
    if state.get("change_approved"):
        prefer_direct = False
    t, a = state["transport"], state["accommodation"]
    variants, direct_unmet = P.build_variants(
        t["outbound"],
        t["inbound"],
        a["offers"],
        prefer_direct=bool(prefer_direct),
        rejected=state.get("rejected_variants") or [],
    )
    if not variants:
        parts = (state.get("reply_parts") or []) + [
            "Nu mai am combinații disponibile diferite de cele deja respinse. Doriți să schimbăm datele, "
            "destinația sau bugetul?"
        ]
        return Command(update={"reply_parts": parts, "stage": "collecting"}, goto="respond")

    budget = req.get("budget")
    batch = await run_tools(
        [
            ToolRequest(
                "calculate_budget",
                {
                    "flight_total": v["outbound"]["total_price"] + v["inbound"]["total_price"],
                    "hotel_total": v["hotel"]["total_price"],
                    "budget": budget,
                    "currency": v["hotel"].get("currency", "EUR"),
                },
                id=f"call_budget_{v['id']}",
            )
            for v in variants
        ],
        "Computing the cost of each variant against the budget.",
    )
    variants = [P.attach_budget(v, batch.results[f"call_budget_{v['id']}"]) for v in variants]
    update = {"variants": variants, "messages": batch.messages}

    if direct_unmet and prefer_direct and not state.get("change_approved"):
        return Command(update=update, goto="present_change")
    if budget is not None and all(not v.get("within_budget") for v in variants) and not state.get("overrun_approved"):
        return Command(update=update, goto="present_overrun")
    return Command(update=update, goto="present_variants")


def _hitl_message(
    state: TripState, kind: str, question: str, options: list[str], body: str, stage: str
) -> dict[str, Any]:
    text = "\n\n".join((state.get("reply_parts") or []) + [body])
    payload = {"kind": kind, "question": question, "options": options, "reply": text}
    return {"messages": [AIMessage(content=text)], "pending_payload": payload, "stage": stage, "reply_parts": []}


async def present_change(state: TripState) -> dict[str, Any]:
    t = state["transport"]
    out_direct = any(o.get("direct") for o in t["outbound"])
    in_direct = any(o.get("direct") for o in t["inbound"])
    which = " și ".join(x for x, ok in (("dus", out_direct), ("întors", in_direct)) if not ok)
    question = "Nu există zboruri directe pentru datele alese. Acceptați zboruri cu escală?"
    body = (
        f"Ați indicat că preferați zboruri directe, dar pentru datele alese nu există zbor direct ({which}). "
        "Aceasta este o schimbare față de cererea inițială, așa că am nevoie de acordul dvs.\n" + question
    )
    return _hitl_message(
        state,
        "approve_change",
        question,
        ["Da, accept zboruri cu escală", "Nu, caut alte date"],
        body,
        "approving_change",
    )


async def approve_change(
    state: TripState,
) -> Command[Literal["build_variants", "check_requirements", "present_change", "respond"]]:
    payload = state["pending_payload"]
    answer = interrupt(payload)
    decision = await decide(answer, payload)
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=answer)],
        "reply_parts": [],
    }
    if decision.action == "approve" or (decision.action == "select" and decision.option_index == 0):
        return Command(update={**update, "change_approved": True}, goto="build_variants")
    if decision.action == "modify" and decision.trip_changes:
        req = merge_request(state["request"], decision.trip_changes.model_dump())
        return Command(update={**update, "request": req}, goto="check_requirements")
    if decision.action in ("reject", "select"):
        return Command(
            update={**update, "stage": "collecting", "reply_parts": ["Bine. Ce alte date de călătorie ați prefera?"]},
            goto="respond",
        )
    return Command(update={**update, "reply_parts": ["Vă rog răspundeți cu da sau nu."]}, goto="present_change")


async def present_overrun(state: TripState) -> dict[str, Any]:
    req = state["request"]
    variants = state["variants"]
    cheapest = min(variants, key=lambda v: v["total"])
    cur = cheapest["currency"]
    question = (
        f"Toate variantele depășesc bugetul de {P.money(req['budget'], cur)}. "
        "Doriți să continuăm cu variante peste buget?"
    )
    body = (
        f"Am căutat zboruri și cazare cu prețuri reale. Cea mai ieftină combinație găsită costă "
        f"{P.money(cheapest['total'], cur)} (zboruri {P.money(cheapest['flight_total'], cur)} + cazare "
        f"{P.money(cheapest['hotel_total'], cur)}), adică cu {P.money(-cheapest['remaining'], cur)} peste bugetul "
        f"de {P.money(req['budget'], cur)}. Nu depășesc bugetul fără acordul dvs.\n{question}\n"
        "Alternativ, puteți modifica bugetul, perioada sau destinația."
    )
    return _hitl_message(
        state,
        "approve_budget_overrun",
        question,
        ["Da, continuă peste buget", "Nu, ajustez cererea"],
        body,
        "approving_overrun",
    )


async def approve_overrun(
    state: TripState,
) -> Command[Literal["present_variants", "check_requirements", "present_overrun", "respond"]]:
    payload = state["pending_payload"]
    answer = interrupt(payload)
    decision = await decide(answer, payload)
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=answer)],
        "reply_parts": [],
    }
    if decision.action == "modify" and decision.trip_changes:
        req = merge_request(state["request"], decision.trip_changes.model_dump())
        return Command(update={**update, "request": req}, goto="check_requirements")
    if decision.action == "approve" or (decision.action == "select" and decision.option_index == 0):
        return Command(update={**update, "overrun_approved": True}, goto="present_variants")
    if decision.action in ("reject", "select"):
        return Command(
            update={
                **update,
                "stage": "collecting",
                "reply_parts": ["Înțeleg. Ce doriți să modificăm: bugetul, perioada sau destinația?"],
            },
            goto="respond",
        )
    return Command(
        update={**update, "reply_parts": ["Vă rog confirmați dacă pot continua peste buget (da/nu)."]},
        goto="present_overrun",
    )


async def present_variants(state: TripState) -> dict[str, Any]:
    req = state["request"]
    variants = state["variants"]
    rec = P.recommended_index(variants)
    adults, children = travellers(req)
    party = f"{adults} adulți" + (f" și {children} copii" if children else "")
    start, end = resolve_dates(req, today())
    dest = state["destination"]
    blocks = "\n\n".join(P.variant_block(i, v, req.get("budget"), i == rec) for i, v in enumerate(variants))
    notes = (state["transport"].get("notes") or []) + (state["accommodation"].get("notes") or [])
    question = "Care variantă vă convine?"
    body = (
        f"Variante pentru {dest['city_name']}, {start:%d.%m.%Y} – {end:%d.%m.%Y}, {party} "
        f"(prețuri obținute acum din căutarea de zboruri și cazare):\n\n{blocks}\n\n"
        f"{P.weather_summary((state.get('activities') or {}).get('weather'))}"
        + (("\n\n" + " ".join(notes)) if notes else "")
        + f"\n\n{question} După alegere vă pregătesc itinerariul detaliat pe zile."
    )
    return _hitl_message(
        state,
        "choose_variant",
        question,
        [P.option_label(i, v) for i, v in enumerate(variants)],
        body,
        "choosing_variant",
    )


async def choose_variant(
    state: TripState,
) -> Command[Literal["generate_itinerary", "build_variants", "check_requirements", "present_variants"]]:
    payload = state["pending_payload"]
    answer = interrupt(payload)
    variants = state["variants"]
    decision = await decide(answer, payload)
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=answer)],
        "reply_parts": [],
    }
    if decision.action == "select" and decision.option_index is not None:
        if decision.trip_changes and any(decision.trip_changes.model_dump(exclude_defaults=True).values()):
            log.info("Variant chosen together with trip changes; changes ignored for the selected variant")
        return Command(
            update={**update, "selected_variant": variants[decision.option_index]}, goto="generate_itinerary"
        )
    if decision.action == "approve":
        return Command(
            update={**update, "selected_variant": variants[P.recommended_index(variants)]}, goto="generate_itinerary"
        )
    if decision.action == "modify" and decision.trip_changes:
        req = merge_request(state["request"], decision.trip_changes.model_dump())
        return Command(update={**update, "request": req}, goto="check_requirements")
    if decision.action == "reject":
        rejected = list(state.get("rejected_variants") or []) + [v["signature"] for v in variants]
        return Command(
            update={**update, "rejected_variants": rejected, "reply_parts": ["Am înțeles, caut alte combinații."]},
            goto="build_variants",
        )
    return Command(
        update={**update, "reply_parts": ["Nu am înțeles ce variantă preferați — indicați numărul ei (1, 2 sau 3)."]},
        goto="present_variants",
    )


# =============================================================== itinerary, final approval, actions


async def generate_itinerary(state: TripState) -> dict[str, Any]:
    req = state["request"]
    v = state["selected_variant"]
    dest = state["destination"]
    start, end = resolve_dates(req, today())
    activities = state.get("activities") or {}
    guide = activities.get("guide") or []
    program, issues, attempts = await P.generate_program(
        v,
        req,
        activities.get("weather"),
        guide,
        [p["text"] for p in state.get("preferences") or []],
        start,
        end,
        dest["city_name"],
    )
    adults, children = travellers(req)
    party = f"{adults} adulți" + (f" + {children} copii" if children else "")
    sections = [
        f"**Itinerariu: {dest['city_name']}, {start:%d.%m.%Y} – {end:%d.%m.%Y}, {party}** (varianta {v['label']})",
        f"- Zbor dus: {P.flight_line(v['outbound'])}\n- Zbor întors: {P.flight_line(v['inbound'])}\n"
        f"- Cazare: {P.hotel_line(v['hotel'])}\n- Buget: {P.budget_sentence(v, req.get('budget'))}",
        P.weather_summary(activities.get("weather")),
        "**Program pe zile:**\n" + program,
    ]
    src = P.sources_block(guide)
    if src:
        sections.append(src)
    text = "\n\n".join(sections)
    # Final self-check of the whole text against the state (prices, dates, booking claims).
    final_issues = P.verify_text(text, v, req.get("budget"), start, end)
    return {
        "itinerary": {"text": text, "verification_issues": issues + final_issues, "attempts": attempts},
    }


async def present_final(state: TripState) -> dict[str, Any]:
    question = "Aprobați acest itinerariu? Îl pot adăuga în calendarul dvs."
    body = (
        state["itinerary"]["text"]
        + "\n\nNotă: nu efectuez rezervări sau plăți; după aprobare pot adăuga călătoria în calendarul dvs.\n"
        + question
    )
    return _hitl_message(
        state,
        "final_approval",
        question,
        ["Aprob și adaug în calendar", "Aprob, fără calendar", "Vreau altă variantă"],
        body,
        "final_approval",
    )


async def final_approval(
    state: TripState,
) -> Command[Literal["finalize", "present_variants", "check_requirements", "present_final"]]:
    payload = state["pending_payload"]
    answer = interrupt(payload)
    decision = await decide(answer, payload)
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=answer)],
        "reply_parts": [],
    }
    if decision.action == "approve" or (decision.action == "select" and decision.option_index == 0):
        return Command(update={**update, "final_decision": "approve_with_calendar"}, goto="finalize")
    if decision.action == "select" and decision.option_index == 1:
        return Command(update={**update, "final_decision": "approve_without_calendar"}, goto="finalize")
    if decision.action == "modify" and decision.trip_changes:
        req = merge_request(state["request"], decision.trip_changes.model_dump())
        return Command(update={**update, "request": req, "selected_variant": None}, goto="check_requirements")
    if decision.action in ("reject", "select"):
        return Command(update={**update, "selected_variant": None}, goto="present_variants")
    return Command(
        update={**update, "reply_parts": ["Vă rog să confirmați: aprobați itinerariul (și adăugarea în calendar)?"]},
        goto="present_final",
    )


async def finalize(state: TripState) -> Command[Literal["respond"]]:
    req = state["request"]
    v = state["selected_variant"]
    dest = state["destination"]
    start, end = resolve_dates(req, today())
    parts: list[str] = []
    update: dict[str, Any] = {"stage": "done"}
    if state.get("final_decision") == "approve_with_calendar":
        batch = await run_tools(
            [
                ToolRequest(
                    "add_calendar_event",
                    {
                        "summary": f"Vacanță {dest['city_name']} "
                        f"({v['outbound']['flight_number']} / {v['inbound']['flight_number']})",
                        "start_date": str(start),
                        "end_date": str(end),
                        "description": f"Hotel: {v['hotel']['name']}. "
                        f"Total estimat: {P.money(v['total'], v['currency'])}.",
                    },
                )
            ],
            "User approved: adding the trip to the calendar.",
        )
        res = next(iter(batch.results.values()))
        update["messages"] = batch.messages
        if res.get("ok"):
            update["calendar_event"] = res
            parts.append(
                f"Gata! Am adăugat călătoria în calendarul dvs. ({start:%d.%m.%Y} – {end:%d.%m.%Y}, "
                f"id eveniment {res.get('event_id')})."
            )
        else:
            parts.append(f"Itinerariul este aprobat, dar nu am reușit să-l adaug în calendar: {res.get('error')}.")
    else:
        parts.append("Gata! Itinerariul este aprobat și pregătit; nu l-am adăugat în calendar.")
    parts.append(
        f"Rezumat: {dest['city_name']}, {start:%d.%m.%Y} – {end:%d.%m.%Y}; {P.budget_sentence(v, req.get('budget'))} "
        "Rețineți că biletele și hotelul nu au fost rezervate sau plătite — acestea se rezervă separat."
    )
    return Command(update={**update, "reply_parts": parts}, goto="respond")


async def respond(state: TripState) -> dict[str, Any]:
    parts = [p for p in (state.get("reply_parts") or []) if p]
    if not parts:
        if state.get("stage") == "done":
            parts = [
                "Planificarea acestei călătorii este finalizată. Dacă doriți modificări (alte date, alt hotel) "
                "sau o nouă vacanță, spuneți-mi."
            ]
        else:
            parts = [
                "Vă pot ajuta să planificați o vacanță: spuneți-mi destinația (sau ce fel de vacanță doriți), "
                "perioada, numărul de persoane, orașul de plecare și bugetul."
            ]
    text = "\n\n".join(parts)
    return {"messages": [AIMessage(content=text)], "last_reply": text, "reply_parts": [], "pending_payload": None}


# =============================================================== graph


def build_graph(checkpointer=None):
    g = StateGraph(TripState)
    for node in (
        start_turn,
        check_requirements,
        resolve_places,
        present_destinations,
        choose_destination,
        delegate,
        build_variants,
        present_change,
        approve_change,
        present_overrun,
        approve_overrun,
        present_variants,
        choose_variant,
        generate_itinerary,
        present_final,
        final_approval,
        finalize,
        respond,
    ):
        g.add_node(node.__name__, node)
    g.add_edge(START, "start_turn")
    g.add_edge("present_destinations", "choose_destination")
    g.add_edge("present_change", "approve_change")
    g.add_edge("present_overrun", "approve_overrun")
    g.add_edge("present_variants", "choose_variant")
    g.add_edge("generate_itinerary", "present_final")
    g.add_edge("present_final", "final_approval")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
