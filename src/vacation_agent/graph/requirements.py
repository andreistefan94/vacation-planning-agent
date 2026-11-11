"""Deterministic requirement handling: merging turn updates, detecting missing slots, deriving dates."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from vacation_agent.graph.state import TripRequest

REQUIRED_SLOTS = ("destination", "origin", "travellers", "period", "budget")

SLOT_LABELS_RO = {
    "destination": "destinația (orașul sau măcar țara/zona dorită)",
    "origin": "orașul de plecare",
    "travellers": "numărul de călători (adulți și copii)",
    "period": "perioada (datele exacte sau luna și durata în zile)",
    "budget": "bugetul total aproximativ",
}


def merge_request(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay the non-empty fields of ``update`` on ``current``.

    Changing the destination clears the destination-specific fields, and giving exact dates
    clears the month/duration assumptions (and vice versa).
    """
    merged = TripRequest(**(current or {})).model_dump()
    if not update:
        return merged
    upd = {k: v for k, v in TripRequest(**update).model_dump().items() if v not in (None, [], "")}
    if "destination" in upd and upd["destination"] != merged.get("destination"):
        for key in ("destination_kind", "destination_city_en", "candidate_cities_en"):
            merged[key] = TripRequest.model_fields[key].get_default(call_default_factory=True)
    if "start_date" in upd and "end_date" not in upd and merged.get("end_date"):
        merged["end_date"] = None
    if "notes" in upd:
        upd["notes"] = list(dict.fromkeys((merged.get("notes") or []) + upd["notes"]))
    merged.update(upd)
    if merged.get("start_date") and merged.get("end_date"):
        try:
            nights = (date.fromisoformat(merged["end_date"]) - date.fromisoformat(merged["start_date"])).days
            if nights > 0:
                merged["duration_days"] = nights
        except ValueError:
            pass
    return merged


def infer_year(month: int, today: date) -> int:
    return today.year if month >= today.month else today.year + 1


def resolve_dates(req: dict[str, Any], today: date) -> tuple[date, date] | None:
    """Exact dates if the user gave them (start+end, or start+duration)."""
    try:
        start = date.fromisoformat(req["start_date"]) if req.get("start_date") else None
        end = date.fromisoformat(req["end_date"]) if req.get("end_date") else None
    except ValueError:
        return None
    if start and end and end > start:
        return start, end
    if start and req.get("duration_days"):
        return start, start + timedelta(days=int(req["duration_days"]))
    return None


def search_window(req: dict[str, Any], today: date) -> tuple[date, date] | None:
    """If only a month (+ duration) is known, the window inside which dates can be assumed."""
    if not req.get("month") or not req.get("duration_days"):
        return None
    month = int(req["month"])
    year = int(req.get("year") or infer_year(month, today))
    first = date(year, month, 1)
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    hint = req.get("period_hint")
    if hint == "early":
        last = min(last, first + timedelta(days=14))
    elif hint == "mid":
        first, last = first + timedelta(days=8), min(last, first + timedelta(days=24))
    elif hint == "late":
        first = first + timedelta(days=15)
    first = max(first, today + timedelta(days=1))
    return first, last


def missing_slots(req: dict[str, Any], today: date) -> list[str]:
    missing = []
    if not req.get("destination") and not req.get("destination_city_en"):
        missing.append("destination")
    if not req.get("origin_city_en"):
        missing.append("origin")
    if not req.get("adults"):
        missing.append("travellers")
    if not resolve_dates(req, today) and not search_window(req, today):
        missing.append("period")
    if req.get("budget") is None:
        missing.append("budget")
    return missing


def pick_free_window(window: tuple[date, date], duration: int, busy_days: set[str]) -> tuple[date, date]:
    """First run of ``duration`` free days (trip days start..end-1) inside the window; else window start."""
    start, last = window
    d = start
    while d + timedelta(days=duration) <= last + timedelta(days=1):
        days = {str(d + timedelta(days=i)) for i in range(duration + 1)}
        if not days & busy_days:
            return d, d + timedelta(days=duration)
        d += timedelta(days=1)
    return start, start + timedelta(days=duration)


def travellers(req: dict[str, Any]) -> tuple[int, int]:
    return int(req.get("adults") or 1), int(req.get("children") or 0)


def describe_request(req: dict[str, Any]) -> str:
    adults, children = travellers(req)
    party = f"{adults} adulți" + (f" + {children} copii" if children else "")
    budget = f"{req['budget']:.0f} {req.get('currency') or 'EUR'}" if req.get("budget") is not None else "—"
    return (
        f"destinație: {req.get('destination_city_en') or req.get('destination') or '—'}; "
        f"plecare din: {req.get('origin_city_en') or '—'}; "
        f"perioada: {req.get('start_date') or req.get('month') or '—'}"
        f"{' → ' + req['end_date'] if req.get('end_date') else ''}; "
        f"durata: {req.get('duration_days') or '—'} zile; călători: {party}; buget: {budget}"
    )
