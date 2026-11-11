"""Tool execution, kept separate from tool selection.

Whoever decides *which* tools to call (an LLM in a sub-agent, or deterministic orchestrator code)
hands a list of ``ToolRequest``s to ``execute_tools``. This module:
- executes independent calls in parallel;
- records the calls as an ``AIMessage(tool_calls=...)`` followed by matching ``ToolMessage``s, so
  the conversation history exposes exactly which tool ran, with what arguments, and what came back;
- retries a failed call once at the agent level (visible in the history), after the HTTP layer's own
  retries and mirror fallback have been exhausted.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

AGENT_LEVEL_RETRIES = 1


@dataclass
class ToolRequest:
    name: str
    args: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class ToolBatchResult:
    messages: list[BaseMessage]
    results: dict[str, Any]  # tool call id of the *original* request -> final result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


async def _invoke(tool: BaseTool, args: dict[str, Any]) -> Any:
    try:
        return await tool.ainvoke(args)
    except Exception as exc:  # noqa: BLE001 - validation or unexpected errors become tool errors
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _failed(result: Any) -> bool:
    return isinstance(result, dict) and result.get("ok") is False


async def execute_tools(
    requests: list[ToolRequest],
    tools: dict[str, BaseTool],
    *,
    agent: str,
    thought: str = "",
) -> ToolBatchResult:
    messages: list[BaseMessage] = []
    final: dict[str, Any] = {}
    pending = list(requests)
    original_id = {r.id: r.id for r in requests}
    attempt = 0

    while pending:
        messages.append(
            AIMessage(
                content=thought if attempt == 0 else f"Retrying {len(pending)} failed tool call(s).",
                tool_calls=[{"id": r.id, "name": r.name, "args": _jsonable(r.args)} for r in pending],
                name=agent,
            )
        )
        results = await asyncio.gather(
            *[
                _invoke(tools[r.name], r.args)
                if r.name in tools
                else asyncio.sleep(0, result={"ok": False, "error": f"Unknown tool {r.name}"})
                for r in pending
            ]
        )
        retry: list[ToolRequest] = []
        for req, result in zip(pending, results, strict=True):
            result = _jsonable(result)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=req.id,
                    name=req.name,
                )
            )
            final[original_id[req.id]] = result
            if _failed(result) and attempt < AGENT_LEVEL_RETRIES:
                again = ToolRequest(name=req.name, args=req.args)
                original_id[again.id] = original_id[req.id]
                retry.append(again)
        pending = retry
        attempt += 1

    return ToolBatchResult(messages=messages, results=final)
