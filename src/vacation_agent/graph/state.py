"""Conversation state (short-term memory), checkpointed per conversation_id."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Stage = Literal[
    "collecting",  # gathering requirements
    "choosing_destination",  # HITL: several candidate destinations
    "searching",  # specialists running
    "approving_overrun",  # HITL: all options exceed the budget
    "approving_change",  # HITL: major change vs. the request (e.g. no direct flight)
    "choosing_variant",  # HITL: pick one of the variants
    "final_approval",  # HITL: approve itinerary + calendar insertion
    "done",
]


class TripRequest(BaseModel):
    """Structured trip requirements. Only what the user actually said; never invented."""

    destination: str | None = Field(None, description="Destination exactly as the user phrased it.")
    destination_kind: Literal["city", "country", "region", "vague"] | None = None
    destination_city_en: str | None = Field(None, description="English city name, only if a specific city.")
    candidate_cities_en: list[str] = Field(
        default_factory=list,
        description="For a country/region/vague destination: 2-4 fitting cities (English names).",
    )
    origin_city_en: str | None = Field(None, description="Departure city in English (e.g. 'Chisinau').")
    start_date: str | None = Field(None, description="YYYY-MM-DD, only if the user gave an exact date.")
    end_date: str | None = Field(None, description="YYYY-MM-DD, only if the user gave an exact date.")
    month: int | None = Field(None, ge=1, le=12)
    year: int | None = None
    duration_days: int | None = Field(None, ge=1, le=60, description="Trip length in days/nights.")
    period_hint: Literal["early", "mid", "late"] | None = None
    adults: int | None = Field(None, ge=1, le=9)
    children: int | None = Field(None, ge=0, le=9)
    budget: float | None = Field(None, ge=0, description="Total budget for the whole trip.")
    currency: str | None = None
    direct_flights_only: bool | None = Field(None, description="Only for THIS trip, if the user asked.")
    notes: list[str] = Field(default_factory=list, description="Other wishes for this trip.")


class TripState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    conversation_id: str

    # requirements / short-term memory
    request: dict[str, Any]
    turn: dict[str, Any]  # analysis of the current user message
    preferences: list[dict[str, Any]]  # long-term preferences loaded for this user
    memory_loaded: bool
    pending_preference: dict[str, Any] | None
    stage: Stage
    origin: dict[str, Any] | None
    destination: dict[str, Any] | None
    destination_options: list[dict[str, Any]]
    dates_assumed: bool

    # specialist results
    transport: dict[str, Any] | None
    accommodation: dict[str, Any] | None
    activities: dict[str, Any] | None
    search_signature: str | None

    # variants & decisions
    variants: list[dict[str, Any]]
    rejected_variants: list[str]
    selected_variant: dict[str, Any] | None
    overrun_approved: bool
    change_approved: bool
    itinerary: dict[str, Any] | None
    calendar_event: dict[str, Any] | None
    final_decision: str | None

    # per-turn output
    pending_payload: dict[str, Any] | None  # HITL interrupt payload {kind, question, options, reply}
    reply_parts: list[str]
    last_reply: str
    replan: bool
