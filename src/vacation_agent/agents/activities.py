"""Activities agent: weather forecast (tool) + destination knowledge from the travel guides (RAG).

It gathers the material for the daily programme; the programme itself is written when the
itinerary is generated, once a variant has been chosen.
"""

from __future__ import annotations

from typing import Any

from vacation_agent.agents.base import AgentRun, react_loop
from vacation_agent.agents.protocol import AgentResult, AgentTask, specialist
from vacation_agent.rag.retriever import RAG_TOOLS
from vacation_agent.tools.runner import ToolRequest
from vacation_agent.tools.travel import weather_tool

TOOLS = {weather_tool.name: weather_tool, **RAG_TOOLS}

SYSTEM = """You are the activities specialist of a travel agency team.
In ONE step, call in parallel:
- get_weather_forecast for the destination coordinates and the trip start date;
- search_travel_guide 2-3 times for the destination city with English queries that help plan the days
  (main sights and districts; getting around / from the airport; and, if children travel, family tips).
Prices, opening hours and weather must never come from the guides.
When done, answer with one short sentence."""


def _collect(run: AgentRun) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    weather, passages, errors = None, {}, []
    for call in run.calls:
        res = call.result if isinstance(call.result, dict) else {}
        if not res.get("ok"):
            errors.append(f"{call.name}: {res.get('error')}")
            continue
        if call.name == "get_weather_forecast":
            weather = res
        elif call.name == "search_travel_guide":
            for r in res.get("results", []):
                passages[(r["section"], r["text"][:80])] = r
    return weather, list(passages.values()), errors


@specialist("ActivitiesAgent")
async def run_activities(task: AgentTask) -> AgentResult:
    i = task.inputs
    city = i["city_name"]
    run = await react_loop(
        "activities",
        SYSTEM,
        f"Task: {task.objective}\nInputs: city={city}, latitude={i['latitude']}, longitude={i['longitude']}, "
        f"start_date={i['start_date']}, days={i['days']}, children={i.get('children', 0)}\n"
        f"Traveller interests/preferences: {'; '.join(task.constraints) or 'none'}",
        TOOLS,
        max_steps=2,
    )
    weather, passages, _ = _collect(run)
    required: list[ToolRequest] = []
    if weather is None:
        required.append(
            ToolRequest(
                "get_weather_forecast",
                {"latitude": i["latitude"], "longitude": i["longitude"], "start_date": i["start_date"]},
            )
        )
    if not any(c.name == "search_travel_guide" for c in run.calls):
        required.append(
            ToolRequest("search_travel_guide", {"query": "main sights and districts to visit", "city": city})
        )
        required.append(
            ToolRequest("search_travel_guide", {"query": "getting around and from the airport", "city": city})
        )
    if required:
        await run.run_requests("activities", required, TOOLS, thought="Required weather/guide lookups (deterministic).")
        weather, passages, _ = _collect(run)

    notes = []
    if weather is None:
        notes.append("Prognoza meteo nu este disponibilă momentan.")
    if not passages:
        notes.append("Ghidurile turistice disponibile nu conțin informații pentru această destinație.")
    return AgentResult(
        task_id=task.task_id,
        agent="activities",
        status="ok" if weather and passages else "partial",
        data={"weather": weather, "guide": passages[:8], "summary": run.final_text},
        notes=notes,
        messages=run.messages,
    )
