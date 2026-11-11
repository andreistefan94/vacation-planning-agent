"""Orchestrator <-> specialist agent protocol.

The orchestrator delegates an ``AgentTask`` and receives an ``AgentResult``. Specialists never talk
to the user and never modify the conversation state directly; the orchestrator aggregates their
results, records their tool calls in the conversation history and decides what happens next.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent: Literal["transport", "accommodation", "activities"]
    objective: str
    inputs: dict[str, Any]
    constraints: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    agent: str
    status: Literal["ok", "partial", "failed"]
    data: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)  # tool calls + results, for the history


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    task = inputs.get("task")
    return task.model_dump() if isinstance(task, AgentTask) else inputs


def _trace_outputs(outputs: Any) -> dict[str, Any]:
    result = outputs.get("output", outputs) if isinstance(outputs, dict) else outputs
    if not isinstance(result, AgentResult):
        return outputs if isinstance(outputs, dict) else {"output": str(outputs)}
    counts = {k: len(v) for k, v in result.data.items() if isinstance(v, list)}
    return {"status": result.status, "notes": result.notes, "result_counts": counts}


def specialist(name: str):
    """Decorator: show a specialist agent as its own named step in LangSmith traces.

    The protocol (AgentTask in, AgentResult summary out) is what appears in the trace; the agent's
    LLM and tool calls are nested under it. No-op when tracing is disabled.
    """
    from langsmith import traceable

    return traceable(
        name=name,
        run_type="chain",
        tags=["specialist"],
        process_inputs=_trace_inputs,
        process_outputs=_trace_outputs,
    )
