"""Tests for Situational Awareness & Duplicate Answer Prevention.

Verifies:
1. Bot does not re-answer a question when a human mentor/staff member already answered it in the channel.
2. Bot does not re-answer when the author is mentioned/tagged by someone in subsequent messages.
3. Bot does not re-answer when the author self-resolves ("never mind", "thanks", "got it").
4. Bot still answers when a question genuinely remains unanswered.
5. Classifier evaluates situational context properly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from ai.classifier import MessageClassifier
from ai.provider import MockProvider
from config import Config
from discord_bot.message_handler import MessageHandler


@pytest.fixture
def handler_and_bot():
    cfg = Config()
    classifier = MessageClassifier(MockProvider())
    generator = MagicMock()
    generator.generate_answer = AsyncMock(return_value=("Official hackathon answer", False))
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])
    handler = MessageHandler(
        classifier=classifier,
        generator=generator,
        retriever=retriever,
        memory=MagicMock(),
        database=MagicMock(),
        config=cfg,
    )
    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"
    bot_user.bot = True

    guild = MagicMock()
    guild.name = "Recursive Hackathon"

    ch_general = MagicMock()
    ch_general.name = "general"
    ch_general.mention = "<#1111>"
    ch_general.parent = None
    ch_general.guild = guild

    ch_ask_mentors = MagicMock()
    ch_ask_mentors.name = "ask-mentors"
    ch_ask_mentors.mention = "<#3333>"
    ch_ask_mentors.parent = None
    ch_ask_mentors.guild = guild

    guild.text_channels = [ch_general, ch_ask_mentors]

    return handler, bot_user, guild, ch_general, ch_ask_mentors


@pytest.mark.asyncio
async def test_catch_up_skips_when_mentor_already_answered_in_channel(handler_and_bot):
    """If a mentor/staff answered the participant while the bot was offline,

    the bot must NOT answer again when it comes online.
    """
    handler, bot_user, guild, ch_general, _ = handler_and_bot

    # 1. Hacker user asks a deadline/PPT question
    hacker_role = MagicMock()
    hacker_role.name = "Hacker"
    hacker_author = MagicMock()
    hacker_author.id = 10001
    hacker_author.name = "anxious_hacker"
    hacker_author.display_name = "Anxious Hacker"
    hacker_author.bot = False
    hacker_author.roles = [hacker_role]
    hacker_author.guild_permissions.administrator = False

    msg_hacker = MagicMock()
    msg_hacker.id = 5001
    msg_hacker.channel = ch_general
    msg_hacker.guild = guild
    msg_hacker.author = hacker_author
    msg_hacker.content = "If we submit the ppt at the very last minute, will that anyhow affect on our selection process?"
    msg_hacker.mentions = []
    msg_hacker.reference = None
    msg_hacker.created_at = datetime(2026, 9, 21, 10, 0, 0, tzinfo=timezone.utc)
    msg_hacker.reply = AsyncMock()

    # 2. Mentor (Core Member / Admin) answers in the channel 5 minutes later without Discord reply button
    mentor_role = MagicMock()
    mentor_role.name = "Core Member"
    mentor_author = MagicMock()
    mentor_author.id = 20002
    mentor_author.name = "lead_mentor"
    mentor_author.display_name = "Lead Mentor"
    mentor_author.bot = False
    mentor_author.roles = [mentor_role]
    mentor_author.guild_permissions.administrator = False

    msg_mentor = MagicMock()
    msg_mentor.id = 5002
    msg_mentor.channel = ch_general
    msg_mentor.guild = guild
    msg_mentor.author = mentor_author
    msg_mentor.content = "No, submitting at the last minute has zero penalty! As long as you submit on Devfolio before the cutoff, it will be evaluated fairly."
    msg_mentor.mentions = []
    msg_mentor.reference = None  # Typed directly in channel, no inline reply
    msg_mentor.created_at = datetime(2026, 9, 21, 10, 5, 0, tzinfo=timezone.utc)
    msg_mentor.reply = AsyncMock()

    # Channel history returns [msg_mentor (newest), msg_hacker (oldest)]
    async def async_history(limit=20):
        yield msg_mentor
        yield msg_hacker

    ch_general.history = MagicMock(return_value=async_history())

    count = await handler.catch_up_unanswered_messages(bot_user, [guild], limit_per_channel=20)

    # Bot must NOT reply to msg_hacker because mentor already answered!
    assert count == 0
    msg_hacker.reply.assert_not_called()


@pytest.mark.asyncio
async def test_catch_up_skips_when_author_mentioned_by_subsequent_user(handler_and_bot):
    """If another user or mentor tagged the author in a subsequent message,

    the question has been addressed.
    """
    handler, bot_user, guild, ch_general, _ = handler_and_bot

    hacker_role = MagicMock()
    hacker_role.name = "Hacker"
    hacker_author = MagicMock()
    hacker_author.id = 10001
    hacker_author.name = "rahul"
    hacker_author.display_name = "Rahul"
    hacker_author.bot = False
    hacker_author.roles = [hacker_role]
    hacker_author.guild_permissions.administrator = False

    msg_q = MagicMock()
    msg_q.id = 6001
    msg_q.channel = ch_general
    msg_q.guild = guild
    msg_q.author = hacker_author
    msg_q.content = "Is registration fee required?"
    msg_q.mentions = []
    msg_q.reference = None
    msg_q.created_at = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)
    msg_q.reply = AsyncMock()

    # Subsequent message tags Rahul
    peer = MagicMock()
    peer.id = 30003
    peer.name = "alex"
    peer.bot = False
    peer.roles = [hacker_role]
    peer.guild_permissions.administrator = False

    msg_ans = MagicMock()
    msg_ans.id = 6002
    msg_ans.channel = ch_general
    msg_ans.guild = guild
    msg_ans.author = peer
    msg_ans.content = "@rahul nope, it is completely free!"
    msg_ans.mentions = [hacker_author]
    msg_ans.reference = None
    msg_ans.created_at = datetime(2026, 9, 21, 12, 2, 0, tzinfo=timezone.utc)
    msg_ans.reply = AsyncMock()

    async def async_history(limit=20):
        yield msg_ans
        yield msg_q

    ch_general.history = MagicMock(return_value=async_history())

    count = await handler.catch_up_unanswered_messages(bot_user, [guild], limit_per_channel=20)
    assert count == 0
    msg_q.reply.assert_not_called()


@pytest.mark.asyncio
async def test_catch_up_skips_when_author_self_resolved(handler_and_bot):
    """If the author sent a follow-up saying 'never mind' or 'got it, thanks',

    the bot should recognize resolution and stay quiet.
    """
    handler, bot_user, guild, ch_general, _ = handler_and_bot

    hacker_role = MagicMock()
    hacker_role.name = "Hacker"
    hacker_author = MagicMock()
    hacker_author.id = 10001
    hacker_author.name = "riya"
    hacker_author.display_name = "Riya"
    hacker_author.bot = False
    hacker_author.roles = [hacker_role]
    hacker_author.guild_permissions.administrator = False

    msg_q = MagicMock()
    msg_q.id = 7001
    msg_q.channel = ch_general
    msg_q.guild = guild
    msg_q.author = hacker_author
    msg_q.content = "Can we use Claude API during the hackathon?"
    msg_q.mentions = []
    msg_q.reference = None
    msg_q.created_at = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
    msg_q.reply = AsyncMock()

    # Riya self-resolves 1 minute later
    msg_ack = MagicMock()
    msg_ack.id = 7002
    msg_ack.channel = ch_general
    msg_ack.guild = guild
    msg_ack.author = hacker_author
    msg_ack.content = "Never mind, saw in #rules that gen AI tools are allowed! Thanks!"
    msg_ack.mentions = []
    msg_ack.reference = None
    msg_ack.created_at = datetime(2026, 9, 21, 14, 1, 0, tzinfo=timezone.utc)
    msg_ack.reply = AsyncMock()

    async def async_history(limit=20):
        yield msg_ack
        yield msg_q

    ch_general.history = MagicMock(return_value=async_history())

    count = await handler.catch_up_unanswered_messages(bot_user, [guild], limit_per_channel=20)
    assert count == 0
    msg_q.reply.assert_not_called()


@pytest.mark.asyncio
async def test_catch_up_answers_genuinely_unanswered_query(handler_and_bot):
    """If a question was asked and NO mentor answered, NO one tagged them,

    and NO self-resolution occurred, the bot MUST answer!
    """
    handler, bot_user, guild, ch_general, _ = handler_and_bot

    # Set generator mock to return an answer
    handler.generator.generate_answer = AsyncMock(return_value=("Registration closes on 25 September.", False))

    hacker_role = MagicMock()
    hacker_role.name = "Hacker"
    hacker_author = MagicMock()
    hacker_author.id = 10001
    hacker_author.name = "sourav"
    hacker_author.display_name = "Sourav"
    hacker_author.bot = False
    hacker_author.roles = [hacker_role]
    hacker_author.guild_permissions.administrator = False

    msg_q = MagicMock()
    msg_q.id = 8001
    msg_q.channel = ch_general
    msg_q.guild = guild
    msg_q.author = hacker_author
    msg_q.content = "When is the registration deadline?"
    msg_q.mentions = []
    msg_q.reference = None
    msg_q.created_at = datetime(2026, 9, 21, 15, 0, 0, tzinfo=timezone.utc)
    msg_q.reply = AsyncMock()

    # Unrelated chatter follows (different topic, not answering Sourav)
    other_user = MagicMock()
    other_user.id = 40004
    other_user.name = "gaming_dude"
    other_user.bot = False
    other_user.roles = [hacker_role]
    other_user.guild_permissions.administrator = False

    msg_chatter = MagicMock()
    msg_chatter.id = 8002
    msg_chatter.channel = ch_general
    msg_chatter.guild = guild
    msg_chatter.author = other_user
    msg_chatter.content = "Anyone wants to play valorant tonight?"
    msg_chatter.mentions = []
    msg_chatter.reference = None
    msg_chatter.created_at = datetime(2026, 9, 21, 15, 5, 0, tzinfo=timezone.utc)
    msg_chatter.reply = AsyncMock()

    async def async_history(limit=20):
        yield msg_chatter
        yield msg_q

    ch_general.history = MagicMock(return_value=async_history())

    count = await handler.catch_up_unanswered_messages(bot_user, [guild], limit_per_channel=20)

    # Bot should answer Sourav's unanswered deadline question
    assert count == 1
    msg_q.reply.assert_called_once()
