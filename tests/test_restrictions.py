"""Unit tests for channel and role restrictions in MessageHandler."""

from unittest.mock import MagicMock
import pytest

from config import Config
from discord_bot.message_handler import MessageHandler


@pytest.fixture
def message_handler():
    cfg = Config()
    handler = MessageHandler(
        classifier=MagicMock(),
        generator=MagicMock(),
        retriever=MagicMock(),
        memory=MagicMock(),
        database=MagicMock(),
        config=cfg,
    )
    return handler


def test_channel_allowed(message_handler):
    # Allowed channels
    ch_general = MagicMock()
    ch_general.name = "general"
    ch_general.parent = None
    assert message_handler._is_channel_allowed(ch_general) is True

    ch_ask_mentors = MagicMock()
    ch_ask_mentors.name = "ask-mentors"
    ch_ask_mentors.parent = None
    assert message_handler._is_channel_allowed(ch_ask_mentors) is True

    # With leading hash
    ch_hash = MagicMock()
    ch_hash.name = "#general"
    ch_hash.parent = None
    assert message_handler._is_channel_allowed(ch_hash) is True

    # Inside thread in general
    thread = MagicMock()
    thread.name = "question-about-api"
    thread.parent = ch_general
    assert message_handler._is_channel_allowed(thread) is True

    # Non-allowed channels
    for name in ["announcements", "rules-and-faq", "schedule", "tech-support", "submission-links", "food-and-swag"]:
        ch = MagicMock()
        ch.name = name
        ch.parent = None
        assert message_handler._is_channel_allowed(ch) is False, f"Channel {name} should be rejected"


def test_author_allowed(message_handler):
    bot_user = MagicMock()
    bot_user.id = 999999

    def make_member(name, roles, is_bot=False, is_admin=False):
        m = MagicMock()
        m.name = name
        m.id = 123456
        m.bot = is_bot
        m.guild_permissions.administrator = is_admin
        role_objs = []
        for r_name in roles:
            r = MagicMock()
            r.name = r_name
            role_objs.append(r)
        m.roles = role_objs
        return m

    # 1. Valid Hacker (participant)
    hacker = make_member("alex", ["Hacker", "@everyone"])
    allowed, reason = message_handler._is_author_allowed(hacker, bot_user)
    assert allowed is True, f"Hacker should be allowed: {reason}"

    # 2. Excluded: Admin
    admin = make_member("sarah", ["Admin", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(admin, bot_user)
    assert allowed is False

    # 3. Excluded: Administrator permission
    admin_perm = make_member("boss", ["Admin", "@everyone"], is_admin=True)
    allowed, _ = message_handler._is_author_allowed(admin_perm, bot_user)
    assert allowed is False

    # 4. Excluded: Moderator role
    mod = make_member("alex_mod", ["Moderator", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(mod, bot_user)
    assert allowed is False

    # 5. Excluded: Core Member
    core_mem = make_member("john", ["Core Member", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(core_mem, bot_user)
    assert allowed is False

    # 6. Excluded: Volunteer
    volunteer = make_member("emma", ["Volunteer", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(volunteer, bot_user)
    assert allowed is False

    # 7. Excluded: Judge
    judge = make_member("dr_smith", ["Judge", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(judge, bot_user)
    assert allowed is False

    # 8. Excluded: Bot flag
    other_bot = make_member("music_bot", ["Hacker"], is_bot=True)
    allowed, _ = message_handler._is_author_allowed(other_bot, bot_user)
    assert allowed is False

    # 9. Excluded: Dyno
    dyno = make_member("Dyno", ["@everyone"])
    allowed, _ = message_handler._is_author_allowed(dyno, bot_user)
    assert allowed is False

    # 10. Excluded: No Hacker role (unverified / spectator)
    spectator = make_member("random_user", ["@everyone"])
    allowed, _ = message_handler._is_author_allowed(spectator, bot_user)
    assert allowed is False

    # 11. Direct mention: Admin pinging @Recur -> Excluded!
    allowed, _ = message_handler._is_author_allowed(admin, bot_user, is_direct_mention=True)
    assert allowed is False

    # 12. Direct mention: Moderator pinging @Recur -> Excluded!
    allowed, _ = message_handler._is_author_allowed(mod, bot_user, is_direct_mention=True)
    assert allowed is False

    # 13. Direct mention: Hacker pinging @Recur -> Allowed!
    allowed, _ = message_handler._is_author_allowed(hacker, bot_user, is_direct_mention=True)
    assert allowed is True


def test_team_finding_channel(message_handler):
    for name in ["find-your-team!", "find-your-team", "#find-your-team!", "🤝-find-your-team!"]:
        ch = MagicMock()
        ch.name = name
        ch.parent = None
        assert message_handler._is_team_finding_channel(ch) is True, f"Channel {name} should be recognized"

    for name in ["general", "ask-mentors", "announcements", "help"]:
        ch = MagicMock()
        ch.name = name
        ch.parent = None
        assert message_handler._is_team_finding_channel(ch) is False


@pytest.mark.asyncio
async def test_team_finding_handler_replies_everyone():
    from unittest.mock import AsyncMock
    from ai.classifier import MessageClassifier
    from ai.provider import MockProvider

    cfg = Config()
    classifier = MessageClassifier(MockProvider())
    handler = MessageHandler(
        classifier=classifier,
        generator=MagicMock(),
        retriever=MagicMock(),
        memory=MagicMock(),
        database=MagicMock(),
        config=cfg,
    )

    bot_user = MagicMock()
    bot_user.id = 999999

    ch = MagicMock()
    ch.name = "find-your-team!"
    ch.parent = None

    # 1. Looking for members message -> Should reply tagging @everyone
    msg = MagicMock()
    msg.channel = ch
    msg.author = MagicMock()
    msg.author.id = 12345
    msg.author.bot = False
    msg.content = "Looking for 2 members for our team. Frontend React, Backend Python. DM me!"
    msg.mentions = []
    msg.reference = None
    msg.reply = AsyncMock()

    handled = await handler._handle_team_finding_message(msg, bot_user)
    assert handled is True
    msg.reply.assert_called_once()
    call_args, call_kwargs = msg.reply.call_args
    assert "@everyone" in call_args[0]
    assert call_kwargs.get("allowed_mentions").everyone is True

    # 2. Duplicate immediate message from same user -> Cooldown active, no second reply
    msg2 = MagicMock()
    msg2.channel = ch
    msg2.author = msg.author
    msg2.content = "Need 1 more teammate"
    msg2.mentions = []
    msg2.reference = None
    msg2.reply = AsyncMock()

    handled2 = await handler._handle_team_finding_message(msg2, bot_user)
    assert handled2 is True
    msg2.reply.assert_not_called()

    # 3. Chatter in team channel -> Ignored, no reply
    msg3 = MagicMock()
    msg3.channel = ch
    msg3.author = MagicMock()
    msg3.author.id = 999
    msg3.author.bot = False
    msg3.content = "lol cool"
    msg3.mentions = []
    msg3.reference = None
    msg3.reply = AsyncMock()

    handled3 = await handler._handle_team_finding_message(msg3, bot_user)
    assert handled3 is True
    msg3.reply.assert_not_called()


@pytest.mark.asyncio
async def test_forward_team_finding_message_from_general():
    from unittest.mock import AsyncMock
    from ai.classifier import MessageClassifier
    from ai.provider import MockProvider

    cfg = Config()
    classifier = MessageClassifier(MockProvider())
    handler = MessageHandler(
        classifier=classifier,
        generator=MagicMock(),
        retriever=MagicMock(),
        memory=MagicMock(),
        database=MagicMock(),
        config=cfg,
    )

    bot_user = MagicMock()
    bot_user.id = 999999

    # Mock channels
    ch_general = MagicMock()
    ch_general.name = "general"
    ch_general.mention = "<#1111>"
    ch_general.parent = None

    ch_team = MagicMock()
    ch_team.name = "find-your-team!"
    ch_team.mention = "<#2222>"
    ch_team.send = AsyncMock()

    # Mock guild with both channels
    guild = MagicMock()
    guild.name = "Recursive Hackathon"
    guild.text_channels = [ch_general, ch_team]

    ch_general.guild = guild

    # Member with Hacker role
    role_hacker = MagicMock()
    role_hacker.name = "Hacker"
    author = MagicMock()
    author.name = "rahul"
    author.id = 55555
    author.bot = False
    author.mention = "<@55555>"
    author.roles = [role_hacker]
    author.guild_permissions.administrator = False

    msg = MagicMock()
    msg.channel = ch_general
    msg.guild = guild
    msg.author = author
    msg.content = "Looking for two members to join my team. Frontend next.js, backend python. DM me!"
    msg.mentions = []
    msg.reference = None
    msg.jump_url = "https://discord.com/channels/1/2/3"
    msg.reply = AsyncMock()

    # Process message in #general
    await handler.handle_message(msg, bot_user)

    # Assert forwarded to #find-your-team! with @everyone
    ch_team.send.assert_called_once()
    send_args, send_kwargs = ch_team.send.call_args
    assert "@everyone" in send_args[0]
    assert "<@55555>" in send_args[0]
    assert "https://discord.com/channels/1/2/3" in send_args[0]
    assert send_kwargs.get("allowed_mentions").everyone is True

    # Assert author received a reply in #general
    msg.reply.assert_called_once()
    reply_args, _ = msg.reply.call_args
    assert "<#2222>" in reply_args[0] or "find-your-team" in reply_args[0]
    assert "<@55555>" in reply_args[0]


@pytest.mark.asyncio
async def test_suppresses_fallback_response_in_ambient_chat():
    from unittest.mock import AsyncMock, MagicMock
    from ai.classifier import MessageClassifier
    from ai.provider import MockProvider

    cfg = Config()
    classifier = MessageClassifier(MockProvider())
    mock_generator = MagicMock()
    # Generator returns fallback (not found in KB)
    mock_generator.generate_answer = AsyncMock(
        return_value=(
            "I couldn't find this information in the official hackathon knowledge base. Please tag @Core Member or @Volunteer for clarification.",
            True,  # was_fallback
        )
    )

    mock_db = MagicMock()
    mock_db.log_unanswered_question = MagicMock()

    handler = MessageHandler(
        classifier=classifier,
        generator=mock_generator,
        retriever=MagicMock(),
        memory=MagicMock(),
        database=mock_db,
        config=cfg,
    )

    bot_user = MagicMock()
    bot_user.id = 999999

    ch_general = MagicMock()
    ch_general.name = "general"
    ch_general.parent = None

    role_hacker = MagicMock()
    role_hacker.name = "Hacker"
    author = MagicMock()
    author.name = "participant1"
    author.id = 11111
    author.bot = False
    author.roles = [role_hacker]
    author.guild_permissions.administrator = False

    msg = MagicMock()
    msg.channel = ch_general
    msg.guild = None
    msg.author = author
    # Ambiguous/ambient question where bot was NOT tagged
    msg.content = "What is the exact room number for the robotics track?"
    msg.mentions = []
    msg.reference = None
    msg.reply = AsyncMock()

    await handler.handle_message(msg, bot_user)

    # Question should be logged for organizers
    mock_db.log_unanswered_question.assert_called_once()

    # BUT reply should be suppressed (bot stays quiet in ambient chat!)
    msg.reply.assert_not_called()


@pytest.mark.asyncio
async def test_replies_fallback_when_directly_mentioned():
    from unittest.mock import AsyncMock, MagicMock
    from ai.classifier import MessageClassifier
    from ai.provider import MockProvider

    cfg = Config()
    classifier = MessageClassifier(MockProvider())
    mock_generator = MagicMock()
    mock_generator.generate_answer = AsyncMock(
        return_value=(
            "I couldn't find this information in the official hackathon knowledge base. Please tag @Core Member or @Volunteer for clarification.",
            True,  # was_fallback
        )
    )

    mock_db = MagicMock()
    handler = MessageHandler(
        classifier=classifier,
        generator=mock_generator,
        retriever=MagicMock(),
        memory=MagicMock(),
        database=mock_db,
        config=cfg,
    )

    bot_user = MagicMock()
    bot_user.id = 999999

    ch_general = MagicMock()
    ch_general.name = "general"
    ch_general.parent = None

    role_hacker = MagicMock()
    role_hacker.name = "Hacker"
    author = MagicMock()
    author.name = "participant1"
    author.id = 11111
    author.bot = False
    author.roles = [role_hacker]
    author.guild_permissions.administrator = False

    msg = MagicMock()
    msg.channel = ch_general
    msg.guild = None
    msg.author = author
    # Directly mentioning the bot: <@999999>
    msg.content = "<@999999> What is the exact room number for robotics?"
    msg.mentions = [bot_user]
    msg.reference = None
    msg.reply = AsyncMock()

    await handler.handle_message(msg, bot_user)

    # When directly mentioned, fallback reply SHOULD be sent to the user
    msg.reply.assert_called_once()
    assert "couldn't find" in msg.reply.call_args[0][0].lower()




