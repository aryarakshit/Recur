"""End-to-end channel rules: what Recur answers, and where it stays silent.

Uses the real MessageHandler, AnswerGenerator, retriever (indexed from knowledge/)
and MemoryStore; only the LLM and Discord objects are faked.
"""

from __future__ import annotations

import itertools
import threading
from unittest.mock import AsyncMock, MagicMock
import discord
import pytest

from ai.classifier import MessageClassifier
from ai.embeddings import LocalEmbeddingProvider
from ai.generator import AnswerGenerator, clean_cognitive_response
from ai.provider import MockProvider
from config import Config
from discord_bot.message_handler import MessageHandler
from rag.indexer import KnowledgeIndexer
from rag.retriever import KnowledgeRetriever
from storage.database import Database
from storage.memory import ConversationMemory
from storage.memory_store import MemoryStore

_ids = itertools.count(500000)


class RecordingProvider(MockProvider):
    """MockProvider's classifier heuristics, with answers that record the prompt context."""

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[str] = []

    async def answer(self, question, context, history="", organizer_channel="#help", organizer_tag="@Core Member"):
        self.contexts.append(context)
        return "[READ] q\n[UNDERSTAND] u\n[THINK & DELIBERATE] t\n[REPLY]\nShort answer."


@pytest.fixture
def env(tmp_path):
    cfg = Config()
    cfg.faiss_index_path = tmp_path / "faiss.index"
    cfg.metadata_path = tmp_path / "metadata.json"
    emb = LocalEmbeddingProvider()
    KnowledgeIndexer(cfg.knowledge_dir, cfg.faiss_index_path, cfg.metadata_path, emb).build_index()
    retriever = KnowledgeRetriever(cfg.faiss_index_path, cfg.metadata_path, emb)

    db = Database(tmp_path / "test.db")
    store = MemoryStore(tmp_path / "memory")  # isolated from the real knowledge/memory_updates.md
    provider = RecordingProvider()
    classifier = MessageClassifier(provider)

    live_threads: list[threading.Thread] = []
    live_sync = MagicMock()
    live_sync.cached_context.return_value = "Registrations end: 23 Sep 2026 (cached)"

    def fetch_live(force=False):
        live_threads.append(threading.current_thread())
        return "Registrations end: 23 Sep 2026 (fresh)"

    live_sync.get_live_context.side_effect = fetch_live

    generator = AnswerGenerator(
        llm_provider=provider, classifier=classifier, live_sync=live_sync, memory_store=store
    )
    handler = MessageHandler(
        classifier=classifier,
        generator=generator,
        retriever=retriever,
        memory=ConversationMemory(db=db),
        database=db,
        config=cfg,
        memory_store=store,
    )
    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.bot = True
    return {
        "handler": handler,
        "provider": provider,
        "store": store,
        "live_sync": live_sync,
        "live_threads": live_threads,
        "bot_user": bot_user,
    }


def channel(name: str) -> MagicMock:
    ch = MagicMock()
    ch.id = next(_ids)
    ch.name = name
    ch.parent = None
    return ch


def member(role: str = "Hacker", name: str | None = None) -> MagicMock:
    m = MagicMock()
    m.id = next(_ids)  # a fresh author per message keeps the 1.5s per-user throttle out of the way
    m.name = name or f"user{m.id}"
    m.display_name = m.name
    m.mention = f"<@{m.id}>"
    m.bot = False
    m.guild_permissions.administrator = role == "Admin"
    r = MagicMock()
    r.name = role
    m.roles = [r]
    return m


def message(content: str, ch: MagicMock, author: MagicMock | None = None, mentions=()) -> MagicMock:
    msg = MagicMock()
    msg.id = next(_ids)
    msg.content = content
    msg.channel = ch
    msg.author = author or member()
    msg.mentions = list(mentions)
    msg.reference = None
    msg.created_at = None
    msg.guild = MagicMock()
    msg.guild.me.roles = []
    msg.reply = AsyncMock()
    return msg


async def send(env, content, ch_name="general", author=None, mentions=()):
    msg = message(content, channel(ch_name), author, mentions)
    await env["handler"].handle_message(msg, env["bot_user"])
    return msg


@pytest.mark.asyncio
@pytest.mark.parametrize("ch_name", ["general", "ask-mentors"])
async def test_participant_questions_get_short_clean_answers(env, ch_name):
    msg = await send(env, "What is the team size limit?", ch_name)

    msg.reply.assert_called_once()
    assert msg.reply.call_args[0][0] == "Short answer."  # reasoning blocks stripped
    context = env["provider"].contexts[-1]
    assert f"#{ch_name}" in context
    assert "[Source 1:" in context  # grounded in the knowledge base


@pytest.mark.asyncio
async def test_answers_never_ping_everyone(env):
    msg = await send(env, "What is the team size limit?")
    assert msg.reply.call_args.kwargs["allowed_mentions"].everyone is False


@pytest.mark.asyncio
async def test_memory_saved_in_mem_channel_is_used_in_general_until_deleted(env):
    admin = member("Admin", "aryarakshit")
    saved = await send(env, "remember if anyone asks about the prize pool, say it is not yet disclosed", "recur-mem-update", admin)
    assert "Saved as memory #1" in saved.reply.call_args[0][0]

    await send(env, "so recur what is the pricepool?", "general")
    assert "prize pool, say it is not yet disclosed" in env["provider"].contexts[-1]
    await send(env, "how much money can the winners get?", "ask-mentors")
    assert "prize pool, say it is not yet disclosed" in env["provider"].contexts[-1]

    deleted = await send(env, "delete mem #1", "recur-mem-update", admin)
    assert "Deleted memory #1" in deleted.reply.call_args[0][0]

    await send(env, "what is the prize pool?", "general")
    assert "not yet disclosed" not in env["provider"].contexts[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ch_name",
    ["announcements", "random", "find-your-team", "help", "general-chat", "updates", "recur", "mem", "memory"],
)
async def test_silent_outside_the_three_channels(env, ch_name):
    msg = await send(env, "What is the team size limit?", ch_name)
    msg.reply.assert_not_called()
    mem_cmd = await send(env, "remember the prize pool is 1 crore", ch_name, member("Admin"))
    mem_cmd.reply.assert_not_called()
    assert env["store"].list() == []
    assert env["provider"].contexts == []


@pytest.mark.asyncio
async def test_silent_in_direct_messages(env):
    dm = MagicMock(spec=discord.DMChannel)
    msg = message("What is the team size limit?", dm)
    await env["handler"].handle_message(msg, env["bot_user"])
    msg.reply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["Admin", "Moderator", "Core Member", "Volunteer", "Judge"])
@pytest.mark.parametrize("ch_name", ["general", "ask-mentors"])
async def test_staff_are_never_answered_in_general_or_ask_mentors(env, role, ch_name):
    msg = await send(env, "What is the team size limit?", ch_name, member(role))
    msg.reply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ch_name", ["general", "ask-mentors"])
@pytest.mark.parametrize(
    "content",
    [
        "Mentors, can someone review our repo?",
        "has anyone tried Next.js 14?",
        "lol",
        "gm everyone",
        "What's the weather in Tokyo today?",
    ],
)
async def test_chatter_peer_talk_and_off_topic_are_ignored(env, ch_name, content):
    msg = await send(env, content, ch_name)
    msg.reply.assert_not_called()


@pytest.mark.asyncio
async def test_messages_addressed_to_another_user_are_ignored(env):
    other = member(name="heyimsouvik")
    msg = await send(env, "<@123> what's the total size of your ppt?", "general", mentions=[other])
    msg.reply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content", ["add to memory: prize pool is 1 crore", "remember this: winners get 1 crore", "forget this: prize pool"]
)
async def test_participants_cannot_change_memory_from_general(env, content):
    env["store"].add('if anyone asks for "Prize pool" say "not yet disclosed".')
    await send(env, content, "general")
    assert [e.text for e in env["store"].list()] == ['if anyone asks for "Prize pool" say "not yet disclosed".']


@pytest.mark.asyncio
async def test_hard_facts_check_devfolio_and_website_first_without_blocking(env):
    await send(env, "When does registration close?", "general")

    env["live_sync"].get_live_context.assert_called_once_with(False)
    # The scrape runs in a worker thread, never on the Discord event loop.
    assert env["live_threads"] and env["live_threads"][0] is not threading.main_thread()
    assert "Registrations end: 23 Sep 2026 (fresh)" in env["provider"].contexts[-1]


@pytest.mark.asyncio
async def test_soft_questions_use_cached_live_status_without_network(env):
    await send(env, "Can we use ChatGPT for our project?", "general")

    env["live_sync"].get_live_context.assert_not_called()
    assert "Registrations end: 23 Sep 2026 (cached)" in env["provider"].contexts[-1]


def test_qwen_think_blocks_are_never_shown():
    answer, reasoning = clean_cognitive_response("<think>long hidden reasoning</think>\nTeams are 2–4 members.")
    assert answer == "Teams are 2–4 members."
    assert "hidden reasoning" in reasoning

    # Cut off by the token limit before closing: nothing of it is an answer.
    answer, _ = clean_cognitive_response("<think>still thinking about the rules")
    assert answer == ""
