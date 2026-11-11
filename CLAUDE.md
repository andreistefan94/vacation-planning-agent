# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language
All the code should be writenn in english, even if I will communciate with you via chat in romanian

## Project

Course final project: AI vacation-planning assistant exposed as a conversational HTTP API. It implements all
five components from the brief (Tools, Human-in-the-Loop, RAG, Memory, Multi-agent), even though only 4 are
required. End-user replies are in Romanian; code, prompts and identifiers are in English.

External contracts (do not change their shapes):
- Assistant API spec: `https://lab-10000-6-3036-svc49-27zuw5t5ra-lm.a.run.app/openapi.json`
- Mock travel APIs (Amadeus / Open-Meteo / Google Calendar style): `.../svc50.../docs`. There is NO
  booking or purchase endpoint, so the agent must never claim to have booked or bought anything.
- Judge + scenarios: `.../svc48.../scenarios`. It checks `hitl_occurred`, `hitl_resumed_not_restarted`,
  `tools_used`, `expected_tools_called` (tool names must contain flight/hotel/weather/preference/location),
  `budget_mentioned`, `retry_on_tool_error`, `preference_stored_after_confirmation`, `no_false_booking_claim`.
- `resources/` holds course-provided material: `corpus/` (Wikivoyage guides), plus the UI and the local runner.

Mock API quirks the code relies on:
- Flight `grandTotal` is per traveller.
- Hotel `price.total` is per night per room.
- `/reference-data/locations` needs an exact English city name or an IATA code.
- Missing parameters return 200 with empty data.

## Commands (PDM)

```
pdm install                      # create .venv and install deps (incl. dev group)
pdm run serve                    # uvicorn with reload on :8000
pdm run test                     # pytest (all tests are offline: fake LLM + respx-mocked HTTP)
pdm run pytest tests/test_api_flows.py::test_full_trip_with_hitl_choices_and_calendar   # single test
pdm run lint / pdm run fmt       # ruff check / ruff format
pdm run fetch-corpus [--force]   # download the 10 Wikivoyage guides as resources/corpus/<city>.json
                                 # (needs WIKIMEDIA_CONTACT in .env - Wikimedia robot policy)
pdm run ingest [--reset]         # chunk + embed resources/corpus into Chroma (local model, ~1.5 min on CPU)
pdm run refresh-corpus [--force] # download + re-index only guides whose Wikivoyage revision changed
pdm run graph                    # print the LangGraph as mermaid
docker build -t vacation-agent . && docker run -p 8000:8000 --env-file .env vacation-agent
```

If `pdm` is not on PATH, use `python -m pdm ...` (system Python). With `.venv` activated, PDM is not
available; run modules directly instead, e.g. `python -m vacation_agent.rag.ingest`, `python -m pytest`.

The corpus format is one JSON file per city: `{city, title, url, license, revision_id, retrieved_at,
sections: [{heading, path, level, text}]}`. `path` (e.g. "Get in > By plane") becomes the chunk's `section`
metadata, which is used in citations. Configuration lives in `.env` (see `.env.example`).
`DATA_DIR` holds the checkpoints, long-term memory and Chroma index (all SQLite or local files).

## Architecture

### Request path
`api/app.py` `ChatService.chat`, thread_id = conversation_id:
- If the thread has a pending `interrupt`, it resumes with `Command(resume=message)`.
- Otherwise it invokes the graph with a new `HumanMessage`.
- `status="awaiting_approval"` exactly when the run stops on an interrupt. The interrupt payload
  `{kind, question, options, reply}` becomes `pending_approval` and `reply`.
- Otherwise `reply` is `state.last_reply`.

### Orchestrator
`graph/orchestrator.py` is a LangGraph `StateGraph`. Routing uses `Command(goto=...)` returned from
nodes. The module docstring contains the flow diagram.
- `start_turn` handles all of the following:
  - loads long-term preferences on the first turn;
  - runs the conversational analysis (`agents/conversational.py`, structured output);
  - merges slots into `request`;
  - handles preference confirmation, booking honesty and RAG questions.
- Missing slots produce a plain clarification and the run ends normally. That is not HITL.
- HITL nodes follow the pattern `present_*` then `choose_*`/`approve_*`:
  - the `present_*` node writes the AIMessage and `pending_payload`;
  - the next node calls `interrupt()`, then interprets the answer with `interpret_decision`.
  - Decisions are select, approve, reject, modify (re-plan via `check_requirements`) or unclear (re-ask).
  - HITL points: destination choice, direct-flight change, budget overrun, variant choice, final
    approval (the calendar insert happens only after it).
- `delegate` runs three specialists in parallel with `asyncio.gather`, using the `agents/protocol.py`
  `AgentTask`/`AgentResult` protocol:
  - transport (flights);
  - accommodation (hotels, availability);
  - activities (weather + RAG).
  Each is a bounded ReAct loop (`agents/base.py`), followed by deterministic guarantees that run any
  required search the LLM skipped.

### Rules enforced in code, not by the LLM
These live in `graph/planning.py` and `graph/requirements.py`:
- Prices and totals come only from tool results. Totals come from the `calculate_budget` tool.
- Only available hotels are used.
- An ambiguous destination never becomes a variant.
- Assumed dates come from free calendar days and are stated to the user.
- `verify_text` rejects invented amounts, out-of-range dates and booking claims. The itinerary is
  regenerated once with that feedback, then falls back to a template.

### Tools
- `tools/travel.py` holds the tools; `tools/client.py` handles OAuth token, tenacity retry and
  mirror fallback.
- Every tool returns `{"ok": bool, ...}` and never raises.
- **All tool execution goes through `tools/runner.execute_tools`.** It does three things:
  - records an `AIMessage(tool_calls)` followed by matching `ToolMessage`s, which is what
    `GET /conversations/{id}` exposes to the judge;
  - runs batches in parallel;
  - retries failed calls once at the agent level, visibly in the history.
- New tools must be registered in `TOOLS` (orchestrator) or in a specialist's tool dict.

### Memory and RAG
- **Short-term memory:** the checkpointed `TripState` (`graph/state.py`).
- **Long-term memory:** `memory/store.py` (SQLite: preferences per user_id, conversation index).
  Permanent preferences are saved only after an explicit "yes".
- **RAG:** `rag/ingest.py` splits on section headings, then by size, with `city/section/source_url/license`
  metadata. Embeddings come from `rag/embeddings.py`: local sentence-transformers
`paraphrase-multilingual-MiniLM-L12-v2` by default, on CPU torch from the pytorch-cpu PDM source, so
Romanian queries work too; Gemini is optional. Each model has its own Chroma collection
(`Settings.collection_name`), so changing the model requires `ingest --reset`.
The corpus updates itself:
- The API server starts `rag/refresh.refresh_loop` as a background task. It runs every
  `CORPUS_REFRESH_DAYS` (default 7; 0 disables it) and needs `WIKIMEDIA_CONTACT`.
- Each run makes one revision check for all 10 pages, then re-downloads and re-indexes only the changed
  cities.
- Chunk ids are `city|rev<id>|n`. New chunks are added before the stale ones are deleted.
- State is kept in `DATA_DIR/corpus_refresh.json` and exposed in `GET /health`.
- Citations use the revision permalink (`...index.php?title=X&oldid=<revision_id>`).

`rag/retriever.py` filters by city, applies a distance threshold (0.68, calibrated on the corpus), and cites
  "Wikivoyage – City, section (CC BY-SA): url". It reports when a city is outside the 10-city corpus.

### Tracing
Observability is LangSmith only; there is no custom trajectory log.
- `config.get_settings()` calls `load_dotenv()` so the `LANGSMITH_*` variables from `.env` reach LangChain.
- `ChatService.chat` names each run `chat_turn`/`resume_turn` and attaches `conversation_id`/`user_id` metadata.
- `tests/conftest.py` forces tracing off.
- The specialists are decorated with `@specialist("TransportAgent")` etc. (`agents/protocol.py`, a LangSmith
  `traceable`). Each appears under `delegate` as its own run: its input is the `AgentTask`, its output is a
  summary of the `AgentResult`, and its LLM and tool calls are nested under it.

### Tests
Tests swap dependencies with `set_llm`, `set_client`, `set_store` and `set_vectorstore`.
- `tests/conftest.py` defines `FakeLLM`: structured outputs are scripted by substring of the user
  message, and `bind_tools` makes no tool calls, so the deterministic plans run.
- `MockTravelApi` (respx) mirrors the mock API's response shapes.
