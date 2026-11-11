"""FastAPI app implementing the conversational API contract."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from vacation_agent.api.history import to_history
from vacation_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationSummary,
    HistoryResponse,
    PendingApproval,
    PreferencesResponse,
)
from vacation_agent.config import get_settings
from vacation_agent.graph.orchestrator import build_graph
from vacation_agent.memory.store import get_store
from vacation_agent.rag.refresh import load_state as corpus_refresh_state
from vacation_agent.rag.refresh import refresh_loop
from vacation_agent.rag.retriever import index_size

log = logging.getLogger(__name__)


class ChatService:
    """Runs one conversational turn on the graph (new input, or resume of a pending interrupt)."""

    def __init__(self, graph):
        self.graph = graph
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def _config(conversation_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": conversation_id}, "recursion_limit": 60}

    async def chat(self, req: ChatRequest) -> ChatResponse:
        store = get_store()
        conversation_id = req.conversation_id or str(uuid.uuid4())
        owner = store.conversation_owner(conversation_id)
        if owner is not None and owner != req.user_id:
            raise HTTPException(status_code=403, detail="This conversation belongs to another user.")

        async with self._locks[conversation_id]:
            config = self._config(conversation_id)
            snapshot = await self.graph.aget_state(config)
            resuming = bool(snapshot.interrupts)
            # LangSmith: one trace per turn, filterable by conversation/user.
            run_config = {
                **config,
                "run_name": "resume_turn" if resuming else "chat_turn",
                "tags": ["hitl_resume"] if resuming else [],
                "metadata": {"conversation_id": conversation_id, "user_id": req.user_id},
            }
            if resuming:
                # Human-in-the-loop: resume the suspended node with the user's decision.
                graph_input: Any = Command(resume=req.message)
            else:
                graph_input = {
                    "messages": [HumanMessage(content=req.message)],
                    "user_id": req.user_id,
                    "conversation_id": conversation_id,
                }
            store.touch_conversation(conversation_id, req.user_id, req.message)
            try:
                await self.graph.ainvoke(graph_input, run_config)
            except HTTPException:
                raise
            except Exception as exc:
                log.exception("Turn failed for conversation %s", conversation_id)
                raise HTTPException(
                    status_code=500, detail=f"The assistant failed to process the message: {exc}"
                ) from exc

            snapshot = await self.graph.aget_state(config)
            if snapshot.interrupts:
                payload = snapshot.interrupts[0].value or {}
                return ChatResponse(
                    conversation_id=conversation_id,
                    user_id=req.user_id,
                    reply=payload.get("reply") or payload.get("question", ""),
                    status="awaiting_approval",
                    pending_approval=PendingApproval(
                        question=payload.get("question", ""), options=payload.get("options", [])
                    ),
                )
            return ChatResponse(
                conversation_id=conversation_id,
                user_id=req.user_id,
                reply=snapshot.values.get("last_reply", ""),
                status="completed",
            )

    async def history(self, conversation_id: str) -> HistoryResponse:
        if get_store().conversation_owner(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Unknown conversation_id")
        snapshot = await self.graph.aget_state(self._config(conversation_id))
        return HistoryResponse(
            conversation_id=conversation_id, messages=to_history(snapshot.values.get("messages", []))
        )


def create_app(service: ChatService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # uvicorn only configures its own loggers; show the app's INFO logs (RAG index, corpus refresh) too.
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
        for noisy in ("httpx", "chromadb", "sentence_transformers", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        if service is not None:
            app.state.service = service
            yield
            return
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        settings = get_settings()
        async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:
            app.state.service = ChatService(build_graph(saver))
            # Load the embedding model + open the index now, not on the first guide question.
            chunks = await asyncio.to_thread(index_size)
            log.info("RAG index: %d chunks in collection %s", chunks, settings.collection_name)
            refresher = asyncio.create_task(refresh_loop(), name="corpus-refresh")
            try:
                yield
            finally:
                refresher.cancel()

    app = FastAPI(
        title="Vacation Planning Assistant",
        version="0.1.0",
        description="Multi-agent vacation planner (tools, HITL, RAG, long-term memory).",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def svc() -> ChatService:
        return app.state.service

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        """The course-provided chat UI (it calls the API with relative URLs, so it is served here)."""
        index = get_settings().ui_dir / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail=f"UI not installed ({index} is missing)")
        return FileResponse(index)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        refresh = corpus_refresh_state()
        return {
            "status": "ok",
            "rag": {k: refresh.get(k) for k in ("last_run", "status", "changed")} if refresh else None,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        return await svc().chat(req)

    @app.get("/conversations/{conversation_id}", response_model=HistoryResponse)
    async def get_history(conversation_id: str) -> HistoryResponse:
        return await svc().history(conversation_id)

    @app.get("/users/{user_id}/conversations", response_model=ConversationListResponse)
    async def list_user_conversations(user_id: str) -> ConversationListResponse:
        return ConversationListResponse(
            user_id=user_id,
            conversations=[ConversationSummary(**c) for c in get_store().list_conversations(user_id)],
        )

    @app.get("/users/{user_id}/preferences", response_model=PreferencesResponse)
    async def get_user_preferences(user_id: str) -> PreferencesResponse:
        return PreferencesResponse(user_id=user_id, preferences=get_store().preference_texts(user_id))

    @app.delete("/users/{user_id}/preferences")
    async def delete_user_preferences(user_id: str) -> dict[str, Any]:
        removed = get_store().clear_preferences(user_id)
        return {"user_id": user_id, "deleted": removed, "preferences": []}

    return app


app = create_app()
