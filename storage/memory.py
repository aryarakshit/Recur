"""Conversation memory management for Hackathon Discord AI Agent.

Maintains sliding recent conversation window per channel/user to enable
follow-up questions without letting conversation history override authoritative knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.database import Database


class ConversationMemory:
    def __init__(self, db: Database, max_history_turns: int = 6) -> None:
        self.db = db
        self.max_history_turns = max_history_turns

    def add_user_message(self, channel_id: str | int, user_id: str | int, message: str) -> None:
        self.db.add_conversation_turn(
            channel_id=channel_id,
            user_id=user_id,
            role="user",
            content=message,
            max_history=self.max_history_turns,
        )

    def add_assistant_message(self, channel_id: str | int, user_id: str | int, message: str) -> None:
        self.db.add_conversation_turn(
            channel_id=channel_id,
            user_id=user_id,
            role="assistant",
            content=message,
            max_history=self.max_history_turns,
        )

    def get_history_context(self, channel_id: str | int, user_id: str | int) -> str:
        """Returns chronological conversation history formatted as a string for LLM context."""
        turns = self.db.get_conversation_history(
            channel_id=channel_id,
            user_id=user_id,
            limit=self.max_history_turns,
        )
        if not turns:
            return ""

        lines = []
        for turn in turns:
            prefix = "Participant" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")

        return "\n".join(lines)
