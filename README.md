# vacation-planning-agent
AI-powered agent for handling requests from users who want to plan, book a vacation

Multi-agent LangGraph assistant (Gemini) implementing tools, human-in-the-loop approvals, RAG over Wikivoyage
guides, long-term memory and an orchestrator with specialist agents, exposed through the course's `/chat` API.

## Quick start

```bash
pip install pdm            # once
pdm install
cp .env.example .env       # set GOOGLE_API_KEY
pdm run ingest             # index resources/corpus (Wikivoyage guides) into Chroma
pdm run serve              # http://localhost:8000/docs
```

Endpoints:
- `POST /chat`
- `GET /conversations/{id}`
- `GET /users/{id}/conversations`
- `GET` and `DELETE /users/{id}/preferences`
- `GET /health`

Tracing: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, plus either `LANGSMITH_AGENT_ID` with
`LANGSMITH_AGENT_ENVIRONMENT`, or `LANGSMITH_PROJECT` (never both: the SDK then drops every trace). Each turn appears
in LangSmith as `chat_turn` or `resume_turn`, with `conversation_id`/`user_id` metadata.

Tests: `pdm run test`. They run offline with a fake LLM and mocked HTTP.

Evaluation with the course runner:
`python runner.py --target http://localhost:8000 --judge <judge url> --label "<name>"`.

Docker: `docker build -t vacation-agent . && docker run -p 8000:8000 --env-file .env vacation-agent`.

RAG sources are Wikivoyage guides (CC BY-SA); answers cite the guide and section they use.
