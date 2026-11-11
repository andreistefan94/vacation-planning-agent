"""Request/response models, mirroring the official assistant API specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="Stable identifier for long-term memory and preferences.")
    conversation_id: str | None = Field(None, description="Omit to start a new conversation.")
    message: str = Field(description="The user's message, or their answer to a pending approval request.")


class PendingApproval(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    user_id: str
    reply: str
    status: Literal["completed", "awaiting_approval"]
    pending_approval: PendingApproval | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: list[HistoryMessage]


class ConversationSummary(BaseModel):
    conversation_id: str
    started_at: str
    last_message_at: str
    preview: str = ""


class ConversationListResponse(BaseModel):
    user_id: str
    conversations: list[ConversationSummary] = Field(default_factory=list)


class PreferencesResponse(BaseModel):
    user_id: str
    preferences: list[str] = Field(default_factory=list)
