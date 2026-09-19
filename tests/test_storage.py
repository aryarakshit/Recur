"""Unit tests for SQLite storage and conversation memory."""

import pytest
from storage.database import Database
from storage.memory import ConversationMemory


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test_hackbot.db"
    return Database(db_path)


def test_log_and_fetch_unanswered_question(db):
    row_id = db.log_unanswered_question(
        question="Can we use quantum computers?",
        channel_id=12345,
        user_id=67890,
        retrieved_sources=["technology-rules.md"],
    )
    assert row_id > 0

    unanswered = db.get_unanswered_questions(limit=5)
    assert len(unanswered) == 1
    assert unanswered[0]["question"] == "Can we use quantum computers?"
    assert unanswered[0]["retrieved_sources"] == ["technology-rules.md"]


def test_conversation_memory_sliding_window(db):
    memory = ConversationMemory(db=db, max_history_turns=4)

    # Add 6 turns (3 user, 3 assistant)
    memory.add_user_message(channel_id=1, user_id=10, message="Hello")
    memory.add_assistant_message(channel_id=1, user_id=10, message="Hi! How can I help?")
    memory.add_user_message(channel_id=1, user_id=10, message="Can teams have 5 members?")
    memory.add_assistant_message(channel_id=1, user_id=10, message="No, maximum team size is 4.")
    memory.add_user_message(channel_id=1, user_id=10, message="What about 3 members?")
    memory.add_assistant_message(channel_id=1, user_id=10, message="Yes, 3 members is allowed.")

    # History context should retain only the last 4 turns
    history = memory.get_history_context(channel_id=1, user_id=10)
    assert "What about 3 members?" in history
    assert "Yes, 3 members is allowed." in history
    assert "Hello" not in history  # First message pruned by sliding window


def test_system_metrics(db):
    db.set_metric("test_counter", 42)
    assert db.get_metric("test_counter") == 42


def test_clear_conversation_cache(db):
    db.add_conversation_turn(channel_id=1, user_id=10, role="user", content="msg 1")
    db.add_conversation_turn(channel_id=1, user_id=10, role="assistant", content="reply 1")
    assert len(db.get_conversation_history(1, 10)) == 2

    cleared = db.clear_conversation_cache(channel_id=1)
    assert cleared == 2
    assert len(db.get_conversation_history(1, 10)) == 0
