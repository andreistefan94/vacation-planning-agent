"""Preference tools (long-term memory) and helpers that turn stored preferences into constraints."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vacation_agent.memory.store import get_store

PREFERENCE_CATEGORIES = [
    "direct_flights",
    "flight_times",
    "hotel_breakfast",
    "hotel_level",
    "travel_party",
    "pace",
    "interests",
    "budget_style",
    "general",
]


class GetPreferencesInput(BaseModel):
    user_id: str


class SavePreferenceInput(BaseModel):
    user_id: str
    text: str = Field(description="The preference, phrased as a short statement about the user.")
    category: str = Field(default="general", description=f"One of {PREFERENCE_CATEGORIES}")


async def get_user_preferences(user_id: str) -> dict[str, Any]:
    """Load the long-term preferences remembered for this user."""
    prefs = get_store().get_preferences(user_id)
    return {"ok": True, "user_id": user_id, "preferences": prefs}


async def save_user_preference(user_id: str, text: str, category: str = "general") -> dict[str, Any]:
    """Persist a long-term preference. Only call after the user explicitly confirmed it should be remembered."""
    if category not in PREFERENCE_CATEGORIES:
        category = "general"
    added = get_store().add_preference(user_id, text, category)
    return {"ok": True, "saved": added, "text": text, "category": category}


get_preferences_tool = StructuredTool.from_function(
    coroutine=get_user_preferences,
    name="get_user_preferences",
    description=(get_user_preferences.__doc__ or "").strip(),
    args_schema=GetPreferencesInput,
)
save_preference_tool = StructuredTool.from_function(
    coroutine=save_user_preference,
    name="save_user_preference",
    description=(save_user_preference.__doc__ or "").strip(),
    args_schema=SavePreferenceInput,
)

MEMORY_TOOLS = {t.name: t for t in (get_preferences_tool, save_preference_tool)}


def prefers_direct_flights(preferences: list[dict]) -> bool:
    for p in preferences:
        text = p.get("text", "").lower()
        if p.get("category") == "direct_flights" or "direct" in text:
            if not any(neg in text for neg in ("nu contează", "nu conteaza", "doesn't matter", "not important")):
                return True
    return False
