"""Deterministic planning logic: variants, presentation texts, itinerary generation and verification.

Guarantees enforced here (not left to the LLM):
- every price comes from tool results; totals come from the budget tool;
- only options whose availability was verified enter a variant;
- the itinerary text is checked against the state before it reaches the user (self-reflection).
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from vacation_agent.llm import get_llm, text_of
from vacation_agent.rag.retriever import format_citation

# ------------------------------------------------------------------ helpers


def money(x: float | None, currency: str = "EUR") -> str:
    return "—" if x is None else f"{x:.2f} {currency}"


def iso_minutes(duration: str) -> int:
    m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", duration or "")
    if not m:
        return 10**6
    d, h, mi = (int(g or 0) for g in m.groups())
    return d * 1440 + h * 60 + mi


def fmt_dt(value: str) -> str:
    # 2026-12-18T09:00:00 -> 18.12.2026 09:00
    try:
        d, t = value.split("T")
        y, mo, da = d.split("-")
        return f"{da}.{mo}.{y} {t[:5]}"
    except ValueError:
        return value


def flight_line(f: dict[str, Any]) -> str:
    stops = "direct" if f.get("direct") else f"{f.get('stops')} escală/escale"
    return (
        f"{f['origin']}→{f['destination']} {fmt_dt(f['departure_at'])} (sosire {fmt_dt(f['arrival_at'])}), "
        f"{f['flight_number']}, {stops} — {money(f['total_price'], f['currency'])} "
        f"({money(f['price_per_person'], f['currency'])}/persoană)"
    )


def hotel_line(h: dict[str, Any]) -> str:
    return (
        f"{h['name']} — cameră {h['room_type']}, {h['nights']} nopți × {h['rooms']} cameră/camere — "
        f"{money(h['total_price'], h['currency'])} ({money(h['price_per_night'], h['currency'])}/noapte/cameră)"
    )


def signature(v: dict[str, Any]) -> str:
    return f"{v['outbound']['flight_number']}|{v['inbound']['flight_number']}|{v['hotel']['hotel_id']}"


# ------------------------------------------------------------------ variants


def select_flights(offers: list[dict[str, Any]], prefer_direct: bool) -> tuple[list[dict[str, Any]], bool]:
    """Return usable offers and whether the direct-flight preference could NOT be met."""
    if not prefer_direct:
        return offers, False
    direct = [o for o in offers if o.get("direct")]
    return (direct, False) if direct else (offers, True)


def build_variants(
    outbound: list[dict[str, Any]],
    inbound: list[dict[str, Any]],
    hotels: list[dict[str, Any]],
    *,
    prefer_direct: bool,
    rejected: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Up to 3 distinct variants (Economic / Echilibrat / Confort). Returns (variants, direct_unmet)."""
    out, out_unmet = select_flights(outbound, prefer_direct)
    inn, in_unmet = select_flights(inbound, prefer_direct)
    hotels = [h for h in hotels if h.get("available")]
    if not out or not inn or not hotels:
        return [], out_unmet or in_unmet

    by_price = sorted(hotels, key=lambda h: h["total_price"])
    cheapest_out = min(out, key=lambda f: f["total_price"])
    cheapest_in = min(inn, key=lambda f: f["total_price"])
    fastest_out = min(out, key=lambda f: (iso_minutes(f["duration"]), f["total_price"]))
    fastest_in = min(inn, key=lambda f: (iso_minutes(f["duration"]), f["total_price"]))
    mid_hotel = next((h for h in by_price if h.get("room_rank", 0) >= 1), by_price[len(by_price) // 2])
    top_rank = max(h.get("room_rank", 0) for h in by_price)
    comfort_hotel = min((h for h in by_price if h.get("room_rank", 0) == top_rank), key=lambda h: h["total_price"])

    candidates = [
        ("Economic", "cele mai mici prețuri găsite", cheapest_out, cheapest_in, by_price[0]),
        ("Echilibrat", "raport bun preț/confort", cheapest_out, cheapest_in, mid_hotel),
        ("Confort", "zboruri mai scurte și cameră superioară", fastest_out, fastest_in, comfort_hotel),
    ]
    variants: list[dict[str, Any]] = []
    seen = set(rejected)
    for label, rationale, fo, fi, h in candidates:
        v = {"label": label, "rationale": rationale, "outbound": fo, "inbound": fi, "hotel": h}
        sig = signature(v)
        if sig in seen:
            continue
        seen.add(sig)
        v["signature"] = sig
        v["id"] = f"V{len(variants) + 1}"
        variants.append(v)
    if len(variants) < 2:
        for h in by_price:
            v = {
                "label": "Alternativ",
                "rationale": "alt hotel disponibil",
                "outbound": cheapest_out,
                "inbound": cheapest_in,
                "hotel": h,
            }
            sig = signature(v)
            if sig not in seen:
                seen.add(sig)
                v["signature"], v["id"] = sig, f"V{len(variants) + 1}"
                variants.append(v)
            if len(variants) >= 2:
                break
    return variants, out_unmet or in_unmet


def attach_budget(variant: dict[str, Any], budget_result: dict[str, Any]) -> dict[str, Any]:
    variant = dict(variant)
    variant["flight_total"] = budget_result.get("flight_total")
    variant["hotel_total"] = budget_result.get("hotel_total")
    variant["total"] = budget_result.get("total")
    variant["remaining"] = budget_result.get("remaining")
    variant["within_budget"] = budget_result.get("within_budget")
    variant["currency"] = budget_result.get("currency", "EUR")
    return variant


def recommended_index(variants: list[dict[str, Any]]) -> int:
    """Most comfortable variant that fits the budget; otherwise the cheapest."""
    within = [i for i, v in enumerate(variants) if v.get("within_budget")]
    if within:
        return max(within, key=lambda i: (variants[i]["hotel"].get("room_rank", 0), -variants[i]["total"]))
    return min(range(len(variants)), key=lambda i: variants[i]["total"])


def budget_sentence(v: dict[str, Any], budget: float | None) -> str:
    if budget is None:
        return f"Total estimat: {money(v['total'], v['currency'])}."
    if v.get("within_budget"):
        return (
            f"Total estimat: {money(v['total'], v['currency'])} — în limita bugetului de "
            f"{money(budget, v['currency'])} (rămân {money(v['remaining'], v['currency'])})."
        )
    return (
        f"Total estimat: {money(v['total'], v['currency'])} — DEPĂȘEȘTE bugetul de {money(budget, v['currency'])} "
        f"cu {money(-v['remaining'], v['currency'])}."
    )


def variant_block(i: int, v: dict[str, Any], budget: float | None, recommended: bool) -> str:
    star = " (recomandată)" if recommended else ""
    return (
        f"**{i + 1}. {v['label']}**{star} — {v['rationale']}\n"
        f"- Zbor dus: {flight_line(v['outbound'])}\n"
        f"- Zbor întors: {flight_line(v['inbound'])}\n"
        f"- Cazare: {hotel_line(v['hotel'])}\n"
        f"- {budget_sentence(v, budget)}"
    )


def option_label(i: int, v: dict[str, Any]) -> str:
    return f"{i + 1}. {v['label']} — {v['hotel']['name']} — {money(v['total'], v['currency'])}"


def weather_summary(weather: dict[str, Any] | None) -> str:
    days = (weather or {}).get("days") or []
    if not days:
        return "Prognoza meteo nu este disponibilă pentru această perioadă."
    tmax = max(d["t_max"] for d in days)
    tmin = min(d["t_min"] for d in days)
    rain = sum(d["precipitation_mm"] for d in days)
    rainy = sum(1 for d in days if d["precipitation_mm"] >= 5)
    return (
        f"Prognoză meteo ({days[0]['date']} – {days[-1]['date']}): între {tmin:.0f}°C și {tmax:.0f}°C, "
        f"{rain:.0f} mm precipitații în total ({rainy} zile cu ploaie semnificativă)."
    )


# ------------------------------------------------------------------ itinerary


ITINERARY_PROMPT = """You write the day-by-day programme of a vacation, in {language}.
Use ONLY the facts below. Strict rules:
- One short paragraph or 2-3 bullets per day, from day 1 (arrival) to the last day (departure).
- Take the flight times into account on the arrival and departure days.
- Adapt each day to the forecast of that date (indoor activities on rainy days).
- Use the numbered guide excerpts for places/neighbourhoods/transport and cite them like [1].
  Do not mention places that are not in the excerpts unless they are generic (e.g. "a walk in the centre").
- Do NOT mention any price, cost, total, or budget. Do NOT claim anything was booked or purchased.
- No title, no introduction, no closing question: just the days.

Facts:
{facts}

Guide excerpts:
{excerpts}"""

_AMOUNT = re.compile(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:EUR|€|euro)", re.I)
_BOOKING_CLAIM = re.compile(
    r"(am rezervat|am cump[aă]rat|rezervare(a)? (a fost )?confirmat|bilete(le)? (au fost )?cump[aă]rat|"
    r"booked|purchased|tickets (were|have been) bought)",
    re.I,
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _parse_amount(raw: str) -> float | None:
    s = raw.replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) <= 2 else s.replace(",", "")
    elif s.count(".") == 1 and len(s.split(".")[-1]) == 3:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def allowed_amounts(v: dict[str, Any], budget: float | None) -> set[float]:
    vals = {
        v.get("total"),
        v.get("flight_total"),
        v.get("hotel_total"),
        budget,
        abs(v["remaining"]) if v.get("remaining") is not None else None,
        v["outbound"]["total_price"],
        v["inbound"]["total_price"],
        v["outbound"]["price_per_person"],
        v["inbound"]["price_per_person"],
        v["hotel"]["total_price"],
        v["hotel"]["price_per_night"],
    }
    return {round(float(x), 2) for x in vals if x is not None}


def verify_text(text: str, v: dict[str, Any], budget: float | None, start: date, end: date) -> list[str]:
    """Self-reflection: return the list of problems found in a generated text (empty = OK)."""
    issues = []
    allowed = allowed_amounts(v, budget)
    for m in _AMOUNT.finditer(text):
        amount = _parse_amount(m.group(1))
        if amount is not None and not any(abs(amount - a) <= 1.0 for a in allowed):
            issues.append(f"amount '{m.group(0)}' does not come from the tool results")
    if _BOOKING_CLAIM.search(text):
        issues.append("the text claims a booking/purchase was made, which never happens")
    for d in _ISO_DATE.findall(text):
        try:
            if not start <= date.fromisoformat(d) <= end:
                issues.append(f"date {d} is outside the trip ({start} – {end})")
        except ValueError:
            issues.append(f"invalid date {d}")
    return issues


def fallback_program(v: dict[str, Any], weather: dict[str, Any] | None, start: date, end: date, city: str) -> str:
    days = {d["date"]: d for d in (weather or {}).get("days", [])}
    lines = []
    n = (end - start).days
    for i in range(n + 1):
        day = date.fromordinal(start.toordinal() + i)
        w = days.get(str(day))
        wtxt = f" (prognoză: {w['t_min']:.0f}–{w['t_max']:.0f}°C, {w['precipitation_mm']:.0f} mm)" if w else ""
        if i == 0:
            what = (
                f"Sosire în {city} ({fmt_dt(v['outbound']['arrival_at'])}), cazare la {v['hotel']['name']}, "
                "plimbare ușoară în zonă."
            )
        elif i == n:
            what = f"Check-out și zborul de întoarcere ({fmt_dt(v['inbound']['departure_at'])})."
        else:
            what = "Zi liberă pentru vizitarea obiectivelor principale ale orașului."
        lines.append(f"- **Ziua {i + 1} ({day.strftime('%d.%m')})**{wtxt}: {what}")
    return "\n".join(lines)


async def generate_program(
    v: dict[str, Any],
    req: dict[str, Any],
    weather: dict[str, Any] | None,
    guide: list[dict[str, Any]],
    preferences: list[str],
    start: date,
    end: date,
    city: str,
) -> tuple[str, list[str], int]:
    """LLM programme + verification loop. Returns (programme, verification issues left, attempts)."""
    facts = {
        "city": city,
        "start_date": str(start),
        "end_date": str(end),
        "outbound_flight": {k: v["outbound"][k] for k in ("flight_number", "departure_at", "arrival_at")},
        "return_flight": {k: v["inbound"][k] for k in ("flight_number", "departure_at", "arrival_at")},
        "hotel": v["hotel"]["name"],
        "travellers": {"adults": req.get("adults"), "children": req.get("children") or 0},
        "forecast": (weather or {}).get("days", []),
        "preferences": preferences + (req.get("notes") or []),
    }
    excerpts = "\n\n".join(f"[{i + 1}] {g['text'][:900]}" for i, g in enumerate(guide)) or "(none)"
    prompt = ITINERARY_PROMPT.format(
        language="Romanian", facts=json.dumps(facts, ensure_ascii=False, indent=1), excerpts=excerpts
    )
    feedback = ""
    issues: list[str] = []
    for attempt in range(1, 3):
        try:
            out = await get_llm().ainvoke(
                [SystemMessage(content=prompt), HumanMessage(content="Write the programme." + feedback)]
            )
            text = text_of(out.content).strip()
        except Exception:  # noqa: BLE001
            break
        issues = verify_text(text, v, None, start, end)
        if not text:
            issues = ["empty programme"]
        if not issues:
            return text, [], attempt
        feedback = "\nYour previous draft was rejected by verification: " + "; ".join(issues) + ". Fix it."
    return fallback_program(v, weather, start, end, city), issues, 3


def sources_block(guide: list[dict[str, Any]]) -> str:
    if not guide:
        return ""
    lines = [f"[{i + 1}] {format_citation(g)}" for i, g in enumerate(guide)]
    return "Surse (ghiduri turistice):\n" + "\n".join(lines)
