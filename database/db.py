"""Database interface for Hackathon Discord AI Agent.

Handles:
- SQLite persistence for unanswered questions
- Sliding conversation history
- System metrics and knowledge base rebuild state
- Cache clearing for /clearcache
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Unanswered questions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unanswered_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    retrieved_sources TEXT,
                    resolved INTEGER DEFAULT 0
                )
            """)

            # 2. Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_channel_user 
                ON conversation_history (channel_id, user_id, timestamp DESC)
            """)

            # 3. Key-value system metrics/metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def log_unanswered_question(
        self,
        question: str,
        channel_id: str | int,
        user_id: str | int,
        retrieved_sources: list[str] | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Logs a participant query that could not be answered from the knowledge base."""
        ts = timestamp if timestamp is not None else time.time()
        sources_json = json.dumps(retrieved_sources or [])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO unanswered_questions (question, channel_id, user_id, timestamp, retrieved_sources)
                VALUES (?, ?, ?, ?, ?)
                """,
                (question.strip(), str(channel_id), str(user_id), ts, sources_json),
            )
            conn.commit()
            last_id = cursor.lastrowid or 0
            logger.info("Logged unanswered question #%d from user %s", last_id, user_id)
            return last_id

    def get_unanswered_questions(self, limit: int = 20, unresolved_only: bool = True) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM unanswered_questions"
            params: list[Any] = []
            if unresolved_only:
                query += " WHERE resolved = 0"
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["retrieved_sources"] = json.loads(item["retrieved_sources"])
                except Exception:
                    item["retrieved_sources"] = []
                results.append(item)
            return results

    def add_conversation_turn(
        self,
        channel_id: str | int,
        user_id: str | int,
        role: str,
        content: str,
        max_history: int = 6,
    ) -> None:
        """Records a user or assistant message and limits history window."""
        ts = time.time()
        cid = str(channel_id)
        uid = str(user_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_history (channel_id, user_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, uid, role, content.strip(), ts),
            )

            # Prune older messages exceeding max_history
            cursor.execute(
                """
                DELETE FROM conversation_history
                WHERE id IN (
                    SELECT id FROM conversation_history
                    WHERE channel_id = ? AND user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (cid, uid, max_history),
            )
            conn.commit()

    def get_conversation_history(
        self,
        channel_id: str | int,
        user_id: str | int,
        limit: int = 6,
    ) -> list[dict[str, str]]:
        cid = str(channel_id)
        uid = str(user_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM conversation_history
                WHERE channel_id = ? AND user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cid, uid, limit),
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_conversation_cache(self, channel_id: str | int | None = None) -> int:
        """Clears conversation history (for /clearcache command)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if channel_id is not None:
                cursor.execute("DELETE FROM conversation_history WHERE channel_id = ?", (str(channel_id),))
            else:
                cursor.execute("DELETE FROM conversation_history")
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info("Cleared %d conversation history rows from cache", deleted_count)
            return deleted_count

    def set_metric(self, key: str, value: Any) -> None:
        val_str = json.dumps(value)
        ts = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO system_metrics (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, val_str, ts),
            )
            conn.commit()

    def get_metric(self, key: str, default: Any = None) -> Any:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_metrics WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except Exception:
                    return row["value"]
            return default
