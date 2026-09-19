"""Unit tests for the two-stage reply decision layer."""

import pytest
from ai.classifier import MessageClassifier
from ai.provider import MockProvider


@pytest.fixture
def classifier():
    provider = MockProvider()
    return MessageClassifier(llm_provider=provider)


@pytest.mark.asyncio
async def test_always_answer_on_mention(classifier):
    should_reply, reason = await classifier.should_reply(
        content="Hello bot",
        is_bot_mentioned=True,
        is_reply_to_bot=False,
    )
    assert should_reply is True
    assert "mentioned" in reason.lower()


@pytest.mark.asyncio
async def test_always_answer_on_reply(classifier):
    should_reply, reason = await classifier.should_reply(
        content="What about python?",
        is_bot_mentioned=False,
        is_reply_to_bot=True,
    )
    assert should_reply is True
    assert "reply" in reason.lower()


@pytest.mark.asyncio
async def test_answers_clear_hackathon_questions_without_mention(classifier):
    test_questions = [
        "when is registration closing?",
        "Can international students participate?",
        "What is the maximum team size?",
        "Where can I find the submission deadline?",
        "What are the judging criteria and prizes?",
    ]
    for q in test_questions:
        should_reply, _ = await classifier.should_reply(
            content=q,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is True, f"Failed to detect hackathon question: '{q}'"


@pytest.mark.asyncio
async def test_ignores_casual_chatter(classifier):
    chatter_messages = [
        "bro look at this \U0001f602",
        "good morning everyone",
        "\U0001f602\U0001f602\U0001f602",
        "nice project!",
        "lol haha",
        "congrats guys!",
        "thanks!",
    ]
    for msg in chatter_messages:
        should_reply, reason = await classifier.should_reply(
            content=msg,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is False, f"Erroneously replied to chatter: '{msg}' (Reason: {reason})"


@pytest.mark.asyncio
async def test_ignores_chatter_mentioning_other_user(classifier):
    should_reply, _ = await classifier.should_reply(
        content="<@123456789> check this",
        is_bot_mentioned=False,
        is_reply_to_bot=False,
    )
    assert should_reply is False
