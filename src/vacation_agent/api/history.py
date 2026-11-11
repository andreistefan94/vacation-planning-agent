"""Map checkpointed LangChain messages to the API's HistoryMessage format."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from vacation_agent.api.schemas import HistoryMessage, ToolCall
from vacation_agent.llm import text_of


def to_history(messages: list[BaseMessage], include_system: bool = False) -> list[HistoryMessage]:
    out: list[HistoryMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(HistoryMessage(role="user", content=text_of(m.content)))
        elif isinstance(m, AIMessage):
            out.append(
                HistoryMessage(
                    role="assistant",
                    content=text_of(m.content),
                    tool_calls=[
                        ToolCall(id=tc["id"], name=tc["name"], args=tc.get("args") or {})
                        for tc in (m.tool_calls or [])
                        if tc.get("id")
                    ],
                )
            )
        elif isinstance(m, ToolMessage):
            out.append(HistoryMessage(role="tool", content=text_of(m.content), tool_call_id=m.tool_call_id))
        elif isinstance(m, SystemMessage) and include_system:
            out.append(HistoryMessage(role="system", content=text_of(m.content)))
    return out
