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
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
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

            # 4. Dynamic memory updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)

            # 5. Query execution & operational metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    latency REAL NOT NULL,
                    token_usage INTEGER DEFAULT 0,
                    was_fallback INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
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
        with self._lock, self._get_connection() as conn:
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
        with self._lock, self._get_connection() as conn:
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
        with self._lock, self._get_connection() as conn:
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
        with self._lock, self._get_connection() as conn:
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

    def log_memory_update(
        self,
        content: str,
        channel_id: str | int,
        user_id: str | int,
        author_name: str,
        timestamp: float | None = None,
    ) -> int:
        """Logs a dynamic memory update provided by organizers."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_updates (content, channel_id, user_id, author_name, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content.strip(), str(channel_id), str(user_id), author_name.strip(), ts),
            )
            conn.commit()
            last_id = cursor.lastrowid or 0
            logger.info("Logged memory update #%d from %s (%s)", last_id, author_name, user_id)
            return last_id

    def get_memory_updates(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieves past dynamic memory updates."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, content, channel_id, user_id, author_name, timestamp
                FROM memory_updates
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "channel_id": row["channel_id"],
                    "user_id": row["user_id"],
                    "author_name": row["author_name"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    def delete_memory_update(self, target_text: str) -> int:
        """Deletes dynamic memory updates matching a target keyword or phrase."""
        pattern = f"%{target_text.strip()}%"
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_updates WHERE content LIKE ?", (pattern,))
            deleted = cursor.rowcount
            conn.commit()
            logger.info("Deleted %d memory update records matching '%s'", deleted, target_text)
            return deleted

    def log_query(
        self,
        question: str,
        channel_id: str | int,
        user_id: str | int,
        latency: float,
        token_usage: int = 0,
        was_fallback: bool = False,
        timestamp: float | None = None,
    ) -> int:
        """Logs query execution details (latency, token usage, fallback status)."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_logs (question, channel_id, user_id, latency, token_usage, was_fallback, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (question.strip(), str(channel_id), str(user_id), float(latency), int(token_usage), int(was_fallback), ts),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_query_stats(self) -> dict[str, Any]:
        """Retrieves aggregated operational metrics for the /stats slash command."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(latency) FROM query_logs")
            row = cursor.fetchone()
            total_queries = row[0] if row and row[0] is not None else 0
            avg_latency = row[1] if row and row[1] is not None else 0.0

            cursor.execute("SELECT COUNT(*) FROM unanswered_questions WHERE resolved = 0")
            unanswered_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM memory_updates")
            memory_updates_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(DISTINCT channel_id || '_' || user_id) FROM conversation_history")
            active_conversations = cursor.fetchone()[0] or 0

            return {
                "total_queries": total_queries,
                "avg_latency": round(float(avg_latency), 3),
                "unanswered_count": unanswered_count,
                "memory_updates_count": memory_updates_count,
                "active_conversations": active_conversations,
            }
