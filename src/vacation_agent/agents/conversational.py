"""Conversational agent: talks to the user.

- extracts requirements from each message (structured output, no invented values);
- detects intents (planning, guide questions, preference statements, booking requests);
- interprets the user's answer to a pending decision (HITL resume value);
- phrases clarification questions and guide answers.
Facts (prices, dates, availability) are never produced here; they come from tools via the state.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from vacation_agent.graph.requirements import SLOT_LABELS_RO
from vacation_agent.graph.state import TripRequest
from vacation_agent.llm import get_llm, text_of
from vacation_agent.memory.preferences import PREFERENCE_CATEGORIES

log = logging.getLogger(__name__)

Intent = Literal[
    "plan_trip",  # wants a vacation planned / gives trip details
    "guide_question",  # descriptive question about a destination (neighbourhoods, transport, visas, safety...)
    "state_preference",  # states a general/habitual preference
    "booking_request",  # asks to book / buy / reserve / pay
    "forget_preferences",  # asks to forget stored preferences
    "show_preferences",  # asks what is remembered about them
    "smalltalk",
    "other",
]


class PreferenceCandidate(BaseModel):
    text: str = Field(description="Short statement in the user's language, e.g. 'Preferă zboruri directe'.")
    category: str = Field(description=f"One of: {', '.join(PREFERENCE_CATEGORIES)}")
    explicitly_permanent: bool = Field(
        description="True only if the user says it applies generally/always/in the future "
        "('mereu', 'întotdeauna', 'de obicei', 'always', 'from now on'). "
        "False for a wish that concerns only this trip."
    )


class TurnAnalysis(BaseModel):
    trip: TripRequest = Field(description="ONLY the trip details stated or changed in the latest user message.")
    intents: list[Intent] = Field(default_factory=list)
    guide_question_en: str | None = Field(
        None, description="If intent guide_question: the question rewritten in English for retrieval."
    )
    guide_city_en: str | None = None
    preference_candidates: list[PreferenceCandidate] = Field(default_factory=list)
    preference_confirmation: Literal["yes", "no", "none"] = Field(
        "none", description="Answer to a pending 'shall I remember this preference?' question, if any."
    )
    language: Literal["ro", "en", "other"] = "ro"


class Decision(BaseModel):
    action: Literal["select", "approve", "reject", "modify", "unclear"]
    option_index: int | None = Field(None, description="0-based index of the chosen option, for 'select'.")
    trip_changes: TripRequest | None = Field(
        None, description="New/changed trip details in the answer (e.g. new dates, a different city)."
    )
    feedback: str | None = Field(None, description="What the user wants changed, if they reject/modify.")


ANALYSIS_PROMPT = """You are the conversational front-end of a vacation planning assistant.
Today is {today}. Analyse ONLY the latest user message (earlier context is given for reference).

Rules:
- Extract trip details only if they are explicitly present in the latest message. Never invent or
  guess destinations, dates, budgets or traveller counts. Leave unknown fields null.
- City names in *_en fields must be the English exonym ("Roma" -> "Rome", "Chișinău" -> "Chisinau",
  "Atena" -> "Athens", "Londra" -> "London").
- destination_kind: "city" for a specific city, "country" for a country (e.g. "Italia"),
  "region" for an area ("Coasta de Azur", "Europa"), "vague" for descriptions ("undeva cald la mare").
  For country/region/vague destinations propose 2-4 well-known cities with airports in candidate_cities_en.
- "o săptămână" = 7 days. Month names -> month number; "începutul lui mai" -> period_hint "early".
- "2 adulți și un copil" -> adults 2, children 1. "pentru mine" -> adults 1.
- budget is the total amount for the trip; currency default EUR.
- direct_flights_only only if the user asks it for THIS trip.
- preference_candidates: habits or general preferences ("prefer mereu zboruri directe",
  "călătoresc de obicei cu copilul"). A wish limited to this trip ("vreau un hotel ieftin de data asta")
  is NOT permanent (explicitly_permanent=false).
- preference_confirmation: only if the assistant just asked whether to remember a preference.
- booking_request: the user asks to actually book/buy/reserve/pay.

Currently known trip details: {known}
Pending preference question: {pending}
Current workflow stage: {stage}"""


DECISION_PROMPT = """The user is answering a decision request from a vacation planning assistant.
Today is {today}.
Decision type: {kind}
Question asked: {question}
Options (0-based): {options}

Classify the answer:
- select: the user picks one option (set option_index). "prima"/"first" -> 0, "a doua" -> 1, or by name.
- approve: yes / agreed / continue / ok (for yes-no approvals; also "continue even if over budget").
- reject: no / none of these / cancel, without new concrete details.
- modify: the user gives new concrete trip details (dates, another city, budget...) -> fill trip_changes
  (English city names, YYYY-MM-DD dates).
- unclear: anything else.
If the user both picks an option AND gives new details (e.g. picks the city and gives dates), use
"select" and also fill trip_changes."""


def _context(messages: list[BaseMessage], n: int = 6) -> str:
    lines = []
    for m in messages[-n:]:
        if m.type in ("human", "ai") and isinstance(m.content, str) and m.content.strip():
            role = "user" if m.type == "human" else "assistant"
            lines.append(f"{role}: {m.content[:600]}")
    return "\n".join(lines)


async def analyze_turn(
    message: str,
    history: list[BaseMessage],
    known: dict[str, Any],
    pending_preference: dict[str, Any] | None,
    stage: str,
    today: date,
) -> TurnAnalysis:
    llm = get_llm().with_structured_output(TurnAnalysis)
    system = ANALYSIS_PROMPT.format(
        today=today.isoformat(),
        known=json.dumps({k: v for k, v in known.items() if v not in (None, [], "")}, ensure_ascii=False),
        pending=json.dumps(pending_preference, ensure_ascii=False) if pending_preference else "none",
        stage=stage,
    )
    prompt = [
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation so far:\n{_context(history)}\n\nLatest user message:\n{message}"),
    ]
    try:
        result = await llm.ainvoke(prompt)
        if isinstance(result, TurnAnalysis):
            return result
        return TurnAnalysis.model_validate(result)
    except Exception:  # noqa: BLE001 - degrade to keyword heuristics instead of failing the turn
        log.exception("Turn analysis failed; using heuristics")
        return heuristic_analysis(message)


_YES = re.compile(r"\b(da|sigur|ok|okay|de acord|accept|aprob|confirm|yes|agreed|continu[ăa]|bine)\b", re.I)
_NO = re.compile(r"\b(nu|refuz|anulează|anuleaza|no|cancel|niciuna)\b", re.I)
_ORDINALS = {
    0: r"\b(prima|primul|first|1|unu|varianta 1|opțiunea 1|optiunea 1)\b",
    1: r"\b(a doua|al doilea|second|2|doi|varianta 2)\b",
    2: r"\b(a treia|al treilea|third|3|trei|varianta 3)\b",
}


def heuristic_analysis(message: str) -> TurnAnalysis:
    intents: list[str] = []
    lower = message.lower()
    if re.search(r"rezerv|cump[aă]r|book|pl[aă]te[sș]te|buy", lower):
        intents.append("booking_request")
    if re.search(r"vacan|excursie|călători|calatori|trip|zile|buget", lower):
        intents.append("plan_trip")
    confirmation = "yes" if _YES.search(lower) else "no" if _NO.search(lower) else "none"
    return TurnAnalysis(trip=TripRequest(), intents=intents or ["other"], preference_confirmation=confirmation)


def heuristic_decision(answer: str, options: list[str]) -> Decision | None:
    lower = answer.lower().strip()
    for idx, opt in enumerate(options):
        if opt and opt.lower() == lower:
            return Decision(action="select", option_index=idx)
    if options:
        for idx, pattern in _ORDINALS.items():
            if idx < len(options) and re.search(pattern, lower):
                return Decision(action="select", option_index=idx)
    if len(lower) <= 40:
        if _NO.search(lower) and not _YES.search(lower):
            return Decision(action="reject")
        if _YES.search(lower):
            return Decision(action="approve")
    return None


async def interpret_decision(answer: str, kind: str, question: str, options: list[str], today: date) -> Decision:
    llm = get_llm().with_structured_output(Decision)
    prompt = [
        SystemMessage(
            content=DECISION_PROMPT.format(
                today=today.isoformat(),
                kind=kind,
                question=question,
                options=json.dumps(options, ensure_ascii=False),
            )
        ),
        HumanMessage(content=answer),
    ]
    try:
        result = await llm.ainvoke(prompt)
        decision = result if isinstance(result, Decision) else Decision.model_validate(result)
        if decision.action == "select" and (
            decision.option_index is None or not 0 <= decision.option_index < max(1, len(options))
        ):
            decision.action = "unclear"
        return decision
    except Exception:  # noqa: BLE001
        log.exception("Decision interpretation failed; using heuristics")
        return heuristic_decision(answer, options) or Decision(action="unclear")


CLARIFY_PROMPT = """You are a friendly travel assistant. Write a short reply in {language} that:
- briefly acknowledges what is already known: {known};
- asks, in one message, for the missing information: {missing}.
Do not propose destinations, prices or dates yourself. No more than 4 sentences. No markdown headings."""


async def write_clarification(known_summary: str, missing: list[str], language: str = "ro") -> str:
    missing_txt = "; ".join(SLOT_LABELS_RO[m] for m in missing)
    fallback = (
        "Ca să vă pot propune variante concrete, mai am nevoie de: "
        + "; ".join(SLOT_LABELS_RO[m] for m in missing)
        + ". Mi le puteți transmite, vă rog?"
    )
    try:
        out = await get_llm().ainvoke(
            [
                SystemMessage(
                    content=CLARIFY_PROMPT.format(
                        language="Romanian" if language != "en" else "English",
                        known=known_summary,
                        missing=missing_txt,
                    )
                ),
                HumanMessage(content="Write the reply."),
            ]
        )
        return text_of(out.content).strip() or fallback
    except Exception:  # noqa: BLE001
        return fallback


GUIDE_PROMPT = """Answer the traveller's question in {language} using ONLY the numbered guide excerpts below.
- Cite every statement with the excerpt number in square brackets, e.g. [1].
- If the excerpts do not contain the answer, say clearly that the travel guides available to you do not
  cover this, and do not answer from general knowledge.
- Never state prices, availability, opening hours or weather from the excerpts: those are dynamic and
  must be checked with live tools.
- Keep it under 150 words.

Excerpts:
{excerpts}"""


async def answer_from_guides(question: str, results: list[dict[str, Any]], language: str = "ro") -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {r['text'][:1200]}" for i, r in enumerate(results))
    out = await get_llm().ainvoke(
        [
            SystemMessage(
                content=GUIDE_PROMPT.format(language="Romanian" if language != "en" else "English", excerpts=excerpts)
            ),
            HumanMessage(content=question),
        ]
    )
    return text_of(out.content).strip()
