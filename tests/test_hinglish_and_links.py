"""Unit and integration tests for Hinglish language support, direct important links, and memory updates."""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock
import pytest
import discord

from ai.classifier import MessageClassifier
from ai.generator import AnswerGenerator
from ai.provider import MockProvider, GroqProvider, SYSTEM_PROMPT, CLASSIFIER_PROMPT
from config import Config
from discord_bot.message_handler import MessageHandler
from rag.models import DocumentChunk, RetrievalResult
from rag.retriever import KnowledgeRetriever, normalize_query_text
from storage.database import Database
from storage.memory_store import MemoryStore

_ids = itertools.count(800000)


@pytest.fixture
def temp_env(tmp_path):
    cfg = Config()
    cfg.knowledge_dir = tmp_path / "knowledge"
    cfg.knowledge_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "test.db"
    db = Database(db_path=db_path)
    return cfg, db, tmp_path


@pytest.fixture
def memory_handler(temp_env):
    cfg, db, _ = temp_env
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
        database=db,
        config=cfg,
    )
    return handler


@pytest.fixture
def bot_user():
    user = MagicMock()
    user.id = 999999
    user.name = "Recur"
    user.bot = True
    return user


def make_channel(name: str = "recur-mem-update") -> MagicMock:
    channel = MagicMock()
    channel.id = 888888
    channel.name = name
    channel.parent = None
    return channel


def make_author(role: str = "Admin", name: str = "aryarakshit") -> MagicMock:
    author = MagicMock()
    author.id = next(_ids)
    author.name = name
    author.display_name = name
    author.bot = False
    r = MagicMock()
    r.name = role
    author.roles = [r]
    return author


def make_message(content: str, channel=None, author=None, reference_to=None, mentions=()) -> MagicMock:
    msg = MagicMock()
    msg.id = next(_ids)
    msg.content = content
    msg.channel = channel or make_channel()
    msg.author = author or make_author()
    msg.guild = MagicMock()
    msg.guild.me.roles = []
    msg.mentions = list(mentions)
    msg.created_at = None
    msg.reply = AsyncMock()
    if reference_to is not None:
        msg.reference = MagicMock()
        msg.reference.message_id = reference_to.id
        msg.reference.resolved = reference_to
    else:
        msg.reference = None
    return msg


def reply_text(msg: MagicMock) -> str:
    msg.reply.assert_called_once()
    return msg.reply.call_args[0][0]


# --- Hinglish Heuristics & Classifier Tests ---

def test_hinglish_question_heuristics():
    classifier = MessageClassifier(MockProvider())
    # Questions in Hinglish should evaluate as question or ambiguous (not chatter)
    assert classifier.evaluate_heuristics("kya solo allowed hai?") is not False
    assert classifier.evaluate_heuristics("bhai ppt submission kab tak karna hai?") is not False
    assert classifier.evaluate_heuristics("bhai prize pool kitna hai?") is not False
    assert classifier.evaluate_heuristics("tu kaun hai?") is True
    assert classifier.evaluate_heuristics("ppt template ka link do") is True
    assert classifier.evaluate_heuristics("website link do na") is True


# --- Direct Important Links Tests ---

@pytest.mark.asyncio
async def test_generator_direct_important_links():
    provider = MockProvider()
    generator = AnswerGenerator(llm_provider=provider)

    # 1. PPT Template Link (English & Hinglish)
    ans1, _ = await generator.generate_answer("where can I get the ppt template?", [])
    assert "https://docs.google.com/presentation/d/1Heaa2d_DUVpFmt4Oo2dKZWUOvAVtXQrn1OnsHCBEWEQ/copy" in ans1

    ans2, _ = await generator.generate_answer("bhai ppt template ka link do", [])
    assert "https://docs.google.com/presentation/d/1Heaa2d_DUVpFmt4Oo2dKZWUOvAVtXQrn1OnsHCBEWEQ/copy" in ans2

    # 2. Devfolio Registration Link
    ans3, _ = await generator.generate_answer("what is the devfolio link?", [])
    assert "https://recursiveacm.devfolio.co" in ans3

    ans4, _ = await generator.generate_answer("devfolio portal ka link do", [])
    assert "https://recursiveacm.devfolio.co" in ans4

    # 3. Official Website Link
    ans5, _ = await generator.generate_answer("what is the official website link?", [])
    assert "https://recursiveacm.in" in ans5

    # 4. Discord Link
    ans6, _ = await generator.generate_answer("discord server link do na", [])
    assert "https://discord.gg/SMYB7tJQf" in ans6


# --- Hinglish Identity & Greetings Fast-Path Tests ---

@pytest.mark.asyncio
async def test_generator_hinglish_identity_and_greetings():
    provider = MockProvider()
    generator = AnswerGenerator(llm_provider=provider)

    ans_id, _ = await generator.generate_answer("tu kaun hai?", [])
    assert "Main Recur hoon" in ans_id

    ans_greet, _ = await generator.generate_answer("namaste recur", [])
    assert "Namaste!" in ans_greet

    ans_checkin, _ = await generator.generate_answer("kaise ho recur", [])
    assert "Main badhiya hoon" in ans_checkin


# --- #recur-mem-update Channel Tests (Add & Delete) ---

@pytest.mark.asyncio
async def test_mem_update_add_and_delete_by_shorthand(memory_handler, temp_env, bot_user):
    cfg, db, _ = temp_env
    admin_author = make_author(role="Admin", name="HeadOrganizer")

    # 1. Admin prompts to remember a thing
    msg1 = make_message("remember the venue for round 2 is Seminar Hall A", author=admin_author)
    await memory_handler.handle_message(msg1, bot_user)

    # Verify response
    assert "Saved as memory #1" in reply_text(msg1)

    # Verify saved in knowledge/memory_updates.md
    content = (cfg.knowledge_dir / "memory_updates.md").read_text(encoding="utf-8")
    assert "## Memory Update #1 by HeadOrganizer" in content
    assert "Seminar Hall A" in content

    # Verify saved in SQLite database
    db_records = db.get_memory_updates()
    assert len(db_records) == 1
    assert db_records[0]["author_name"] == "HeadOrganizer"
    assert "Seminar Hall A" in db_records[0]["content"]

    # 2. Add second memory in Hinglish
    msg2 = make_message("yaad rakhna ki lunch coupons registration desk par milenge", author=admin_author)
    await memory_handler.handle_message(msg2, bot_user)
    assert "Saved as memory #2" in reply_text(msg2)
    assert len(db.get_memory_updates()) == 2

    # 3. Admin deletes memory #1 using shorthand "delete #1"
    msg_del1 = make_message("delete #1", author=admin_author)
    await memory_handler.handle_message(msg_del1, bot_user)
    assert "Deleted memory #1" in reply_text(msg_del1)

    # Verify memory #1 is removed from file and DB
    updated_content = (cfg.knowledge_dir / "memory_updates.md").read_text(encoding="utf-8")
    assert "Seminar Hall A" not in updated_content
    assert "lunch coupons" in updated_content
    assert [e.id for e in memory_handler.memory_store.list()] == [2]
    assert len(db.get_memory_updates()) == 1

    # 4. Admin deletes memory #2 using "del #2"
    msg_del2 = make_message("del #2", author=admin_author)
    await memory_handler.handle_message(msg_del2, bot_user)
    assert "Deleted memory #2" in reply_text(msg_del2)
    assert len(memory_handler.memory_store.list()) == 0
    assert len(db.get_memory_updates()) == 0


@pytest.mark.asyncio
async def test_mem_update_delete_by_keyword_and_number(memory_handler, temp_env, bot_user):
    cfg, db, _ = temp_env
    admin_author = make_author(role="Admin")

    # Add memory
    msg_add = make_message("update mem Wi-Fi SSID is HackathonGuest", author=admin_author)
    await memory_handler.handle_message(msg_add, bot_user)
    assert "Saved as memory #1" in reply_text(msg_add)

    # Delete via "delete memory 1" (without # symbol)
    msg_del = make_message("delete memory 1", author=admin_author)
    await memory_handler.handle_message(msg_del, bot_user)
    assert "Deleted memory #1" in reply_text(msg_del)
    assert len(memory_handler.memory_store.list()) == 0
