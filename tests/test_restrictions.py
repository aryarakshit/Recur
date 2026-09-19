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

    # 2. Allowed: Admin
    admin = make_member("sarah", ["Admin", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(admin, bot_user)
    assert allowed is True

    # 3. Allowed: Administrator permission
    admin_perm = make_member("boss", ["Admin", "@everyone"], is_admin=True)
    allowed, _ = message_handler._is_author_allowed(admin_perm, bot_user)
    assert allowed is True

    # 4. Excluded: Core Member
    core_mem = make_member("john", ["Core Member", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(core_mem, bot_user)
    assert allowed is False

    # 5. Excluded: Volunteer
    volunteer = make_member("emma", ["Volunteer", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(volunteer, bot_user)
    assert allowed is False

    # 6. Excluded: Judge
    judge = make_member("dr_smith", ["Judge", "@everyone"])
    allowed, _ = message_handler._is_author_allowed(judge, bot_user)
    assert allowed is False

    # 7. Excluded: Bot flag
    other_bot = make_member("music_bot", ["Hacker"], is_bot=True)
    allowed, _ = message_handler._is_author_allowed(other_bot, bot_user)
    assert allowed is False

    # 8. Excluded: Dyno
    dyno = make_member("Dyno", ["@everyone"])
    allowed, _ = message_handler._is_author_allowed(dyno, bot_user)
    assert allowed is False

    # 9. Excluded: No Hacker role (unverified / spectator)
    spectator = make_member("random_user", ["@everyone"])
    allowed, _ = message_handler._is_author_allowed(spectator, bot_user)
    assert allowed is False

    # 10. Direct mention: Admin directly pinging @Recur -> Allowed!
    allowed, _ = message_handler._is_author_allowed(admin, bot_user, is_direct_mention=True)
    assert allowed is True
