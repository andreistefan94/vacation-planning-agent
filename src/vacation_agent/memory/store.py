"""Long-term memory (preferences per user) and the conversation index, in SQLite.

Short-term memory (the state of the current planning) lives in the LangGraph checkpointer and is
keyed by conversation_id; this store is keyed by user_id and survives across conversations.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from vacation_agent.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    created_at TEXT NOT NULL,
    UNIQUE(user_id, text)
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    preview TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, last_message_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    def __init__(self, path: Path | str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- preferences
    def get_preferences(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT text, category, created_at FROM preferences WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def preference_texts(self, user_id: str) -> list[str]:
        return [p["text"] for p in self.get_preferences(user_id)]

    def add_preference(self, user_id: str, text: str, category: str = "general") -> bool:
        """Add a preference. A newer preference of the same category replaces the older one."""
        text = text.strip()
        if not text:
            return False
        with self._lock:
            if category != "general":
                self._conn.execute(
                    "DELETE FROM preferences WHERE user_id=? AND category=? AND text<>?",
                    (user_id, category, text),
                )
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO preferences(user_id, text, category, created_at) VALUES (?,?,?,?)",
                (user_id, text, category, _now()),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def remove_preference(self, user_id: str, text: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM preferences WHERE user_id=? AND text=?", (user_id, text))
            self._conn.commit()
            return cur.rowcount > 0

    def clear_preferences(self, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM preferences WHERE user_id=?", (user_id,))
            self._conn.commit()
            return cur.rowcount

    # ---------------- conversation index
    def touch_conversation(self, conversation_id: str, user_id: str, first_message: str) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversations(conversation_id, user_id, started_at, last_message_at, preview)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(conversation_id) DO UPDATE SET last_message_at=excluded.last_message_at""",
                (conversation_id, user_id, now, now, first_message[:80]),
            )
            self._conn.commit()

    def conversation_owner(self, conversation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
        return row["user_id"] if row else None

    def list_conversations(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT conversation_id, started_at, last_message_at, preview FROM conversations
                   WHERE user_id=? ORDER BY last_message_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]


_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(get_settings().memory_db)
    return _store


def set_store(store: MemoryStore | None) -> None:
    global _store
    _store = store
