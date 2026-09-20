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
        "is it free?",
        "when does it start?",
        "what are the tracks?",
        "what is the campus address?",
        "is food provided?",
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
        "Hi guys",
        "hello guys",
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


@pytest.mark.asyncio
async def test_ignores_teammate_recruitment_and_lfg(classifier):
    teammate_searches = [
        # User's exact message from screenshot
        "I am finding teamates",
        "finding teammates",
        "seeking teamates",
        "find teamates",
        # User's exact recruitment message
        "Hi guys ,I am looking for two members to join my team Available Domains-applied ai(crew- ai) Frontend -next.js If anyone interested kindly reply or dm",
        "Looking for 1 frontend dev to join our team",
        "Hey everyone, looking for 2 members for AI track, DM me if interested",
        "Need a backend developer for our team, ping me",
        "Anyone interested to join our team? We have 1 slot open",
        "Looking to join a team, skilled in Python and React",
        "If anyone interested kindly reply or dm",
        "Join my team, we need 1 more member",
    ]
    for msg in teammate_searches:
        assert classifier.is_teammate_search(msg) is True, f"Failed to detect teammate search: '{msg}'"
        should_reply, reason = await classifier.should_reply(
            content=msg,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is False, f"Erroneously replied to teammate search: '{msg}' (Reason: {reason})"


@pytest.mark.asyncio
async def test_ignores_messages_addressed_to_mentors_and_staff(classifier):
    # Actual user/participant messages addressed to human staff
    mentor_messages = [
        "Mentors, as today is the last day of submission. Me and my team are still working on the prototype of the idea we've been trying to build and would submit it with the ppt itself. So if we submit the ppt at the very last minute, will that anyhow affect on our selection process? Please do let us know.",
        "Mentors: can someone please review our circuit diagram?",
        "Hey mentors, are you available for a quick doubt?",
        "Hi mentor, quick question about our database structure",
        "Judges, will we be presenting on our laptops or the lab PCs?",
        "Core team, when will dinner be served tonight?",
        "Organizers, where can we get the WiFi credentials?",
        "Can any mentor check our backend repo?",
        "Sir, can you please approve our team?",
        "Anyone from the core team available?",
    ]
    for msg in mentor_messages:
        should_reply, reason = await classifier.should_reply(
            content=msg,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is False, f"Erroneously replied to message addressed to humans: '{msg}' (Reason: {reason})"
        assert "human" in reason.lower() or "filter" in reason.lower()


@pytest.mark.asyncio
async def test_answers_when_bot_is_mentioned_even_if_mentors_addressed(classifier):
    # If the user explicitly mentions @Recur, the bot should always answer
    msg = "Mentors, as today is the last day of submission. If we submit at the very last minute, will it affect our selection?"
    should_reply, reason = await classifier.should_reply(
        content=msg,
        is_bot_mentioned=True,
        is_reply_to_bot=False,
    )
    assert should_reply is True
    assert "mentioned" in reason.lower()


@pytest.mark.asyncio
async def test_does_not_ignore_objective_questions_about_mentors(classifier):
    # Objective questions about mentors/judging rules should NOT be ignored as addressed to humans
    rule_questions = [
        "Who are the mentors?",
        "What are the judging criteria?",
        "Can mentors participate in the hackathon?",
        "Are mentors provided during the hackathon?",
        "Will there be judges for each track?",
    ]
    for q in rule_questions:
        assert classifier.is_addressed_to_human(q) is False, f"Question falsely classified as addressed to human: '{q}'"
        should_reply, reason = await classifier.should_reply(
            content=q,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is True, f"Failed to answer objective question: '{q}' (Reason: {reason})"


@pytest.mark.asyncio
async def test_ignores_peer_conversation(classifier):
    peer_messages = [
        "Hey guys, which library are you using for graphs?",
        "Guys, what do you think of this architecture?",
        "Has anyone tried deploying on Render?",
        "Is anyone else experiencing lag on Devfolio?",
        "What tech stack are you guys using?",
        "How is everyone doing with their projects?",
        "Anyone want to test our API?",
        "Is it just me or is the venue cold?",
    ]
    for msg in peer_messages:
        assert classifier.is_peer_conversation(msg) is True, f"Failed to detect peer conversation: '{msg}'"
        should_reply, reason = await classifier.should_reply(
            content=msg,
            is_bot_mentioned=False,
            is_reply_to_bot=False,
        )
        assert should_reply is False, f"Erroneously replied to peer conversation: '{msg}' (Reason: {reason})"


@pytest.mark.asyncio
async def test_peer_conversation_replies_when_bot_mentioned(classifier):
    msg = "Hey guys, which tech stack can we use for the AI track?"
    should_reply, reason = await classifier.should_reply(
        content=msg,
        is_bot_mentioned=True,
        is_reply_to_bot=False,
    )
    assert should_reply is True
    assert "mentioned" in reason.lower()


