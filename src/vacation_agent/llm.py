"""LLM factory (Gemini). Tests swap the model via ``set_llm``."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from vacation_agent.config import get_settings

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        s = get_settings()
        _llm = ChatGoogleGenerativeAI(
            model=s.gemini_model,
            temperature=s.llm_temperature,
            google_api_key=s.google_api_key or None,
            max_retries=3,
            timeout=60,
        )
    return _llm


def set_llm(llm: BaseChatModel | None) -> None:
    global _llm
    _llm = llm


def text_of(content: Any) -> str:
    """Message content as plain text (Gemini may return a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)
