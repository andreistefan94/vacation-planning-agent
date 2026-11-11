"""Generic ReAct loop for the specialist agents.

The LLM selects tools (with arguments); ``execute_tools`` runs them (in parallel when the model
issues several calls in one step). The loop is bounded, and each specialist adds deterministic
guarantees afterwards (required searches are executed by code if the model skipped them).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from vacation_agent.llm import get_llm, text_of
from vacation_agent.tools.runner import ToolRequest, execute_tools

log = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: Any


@dataclass
class AgentRun:
    messages: list[BaseMessage] = field(default_factory=list)
    calls: list[ToolCallRecord] = field(default_factory=list)
    final_text: str = ""
    llm_failed: bool = False

    async def run_requests(
        self, agent: str, requests: list[ToolRequest], tools: dict[str, BaseTool], thought: str = ""
    ):
        batch = await execute_tools(requests, tools, agent=agent, thought=thought)
        self.messages.extend(batch.messages)
        for req in requests:
            self.calls.append(ToolCallRecord(req.name, req.args, batch.results[req.id]))
        return batch


async def react_loop(agent: str, system: str, task: str, tools: dict[str, BaseTool], max_steps: int = 3) -> AgentRun:
    run = AgentRun()
    try:
        llm = get_llm().bind_tools(list(tools.values()))
    except Exception:  # noqa: BLE001 - e.g. a fake model without tool support
        run.llm_failed = True
        return run
    convo: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=task)]
    for _ in range(max_steps):
        try:
            ai = await llm.ainvoke(convo)
        except Exception:  # noqa: BLE001
            log.exception("%s agent LLM call failed", agent)
            run.llm_failed = True
            break
        if not getattr(ai, "tool_calls", None):
            run.final_text = text_of(ai.content)
            break
        requests = [
            ToolRequest(name=tc["name"], args=tc.get("args") or {}, **({"id": tc["id"]} if tc.get("id") else {}))
            for tc in ai.tool_calls
        ]
        batch = await run.run_requests(agent, requests, tools, thought=text_of(ai.content))
        # Feed back the model's own message (keeps provider metadata such as thought signatures)
        # and the final result of each call (after any agent-level retry).
        convo.append(ai)
        convo.extend(
            ToolMessage(
                content=json.dumps(batch.results[r.id], ensure_ascii=False, default=str)[:6000],
                tool_call_id=r.id,
                name=r.name,
            )
            for r in requests
        )
    return run
