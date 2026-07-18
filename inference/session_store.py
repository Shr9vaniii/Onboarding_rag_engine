"""SQLite session storage for conversational RAG."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "enterprise_data" / "sessions.db"
MAX_PAST_TOPICS = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    active_topic: str
    topic_summary: str
    entities: list[str]
    created_at: str
    updated_at: str
    past_topics: list[str]


@dataclass
class Turn:
    role: str
    content: str
    created_at: str


class SessionStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate_db(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "past_topics" not in columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN past_topics TEXT NOT NULL DEFAULT '[]'"
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    active_topic TEXT NOT NULL DEFAULT '',
                    topic_summary TEXT NOT NULL DEFAULT '',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    past_topics TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                """
            )
            self._migrate_db(conn)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, active_topic, topic_summary, entities_json,
                    created_at, updated_at, past_topics
                ) VALUES (?, '', '', '[]', ?, ?, '[]')
                """,
                (session_id, now, now),
            )
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        past_topics: list[str] = []
        try:
            past_topics = json.loads(row["past_topics"] or "[]")
        except (KeyError, json.JSONDecodeError, TypeError):
            past_topics = []
        return SessionRecord(
            session_id=row["session_id"],
            active_topic=row["active_topic"] or "",
            topic_summary=row["topic_summary"] or "",
            entities=json.loads(row["entities_json"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            past_topics=past_topics,
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)

    def get_recent_turns(self, session_id: str, *, limit: int = 10) -> list[Turn]:
        """Return the last N messages (user + assistant), oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        rows = list(reversed(rows))
        return [
            Turn(role=row["role"], content=row["content"], created_at=row["created_at"])
            for row in rows
        ]

    def get_recent_exchanges(self, session_id: str, *, max_exchanges: int = 3) -> list[Turn]:
        """Return the last N complete user+assistant exchanges, oldest first."""
        fetch_limit = max_exchanges * 2 + 1
        turns = self.get_recent_turns(session_id, limit=fetch_limit)
        if turns and turns[-1].role == "user":
            turns = turns[:-1]
        if len(turns) > max_exchanges * 2:
            turns = turns[-(max_exchanges * 2) :]
        return turns

    def update_session_context(
        self,
        session_id: str,
        *,
        active_topic: str | None = None,
        topic_summary: str | None = None,
        entities: list[str] | None = None,
        past_topics: list[str] | None = None,
    ) -> None:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        active_topic = session.active_topic if active_topic is None else active_topic
        topic_summary = session.topic_summary if topic_summary is None else topic_summary
        entities = session.entities if entities is None else entities
        past_topics = session.past_topics if past_topics is None else past_topics

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET active_topic = ?, topic_summary = ?, entities_json = ?,
                    past_topics = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    active_topic,
                    topic_summary,
                    json.dumps(entities),
                    json.dumps(past_topics[-MAX_PAST_TOPICS:]),
                    _utc_now(),
                    session_id,
                ),
            )

    def archive_active_topic(self, session_id: str) -> list[str]:
        """Move active_topic to past_topics and clear active topic fields."""
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        past_topics = list(session.past_topics)
        if session.active_topic:
            past_topics.append(session.active_topic)
        past_topics = past_topics[-MAX_PAST_TOPICS:]

        self.update_session_context(
            session_id,
            active_topic="",
            topic_summary="",
            entities=[],
            past_topics=past_topics,
        )
        return past_topics
