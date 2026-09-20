"""Unit and integration tests for dynamic memory updates via #recur-mem-update."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ai.classifier import MessageClassifier
from ai.provider import MockProvider
from config import Config
from discord_bot.message_handler import MessageHandler
from storage.database import Database


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
    indexer = MagicMock()
    indexer.build_index = MagicMock(return_value={"chunk_count": 12, "time_taken": 0.05})

    handler = MessageHandler(
        classifier=classifier,
        generator=generator,
        retriever=retriever,
        memory=MagicMock(),
        database=db,
        config=cfg,
        indexer=indexer,
    )
    return handler


def test_extract_memory_update_explicit_commands(memory_handler):
    # 1. User says "@recur add this info in your memory .. The closing ceremony starts at 5 PM."
    text1 = "add this info in your memory .. The closing ceremony starts at 5 PM on Sunday."
    payload1 = memory_handler._extract_memory_update(text1, is_memory_channel=False)
    assert payload1 == "The closing ceremony starts at 5 PM on Sunday."

    # 2. "add to memory: Hackathon winners receive $5000 USD."
    text2 = "add to memory: Hackathon winners receive $5000 USD."
    payload2 = memory_handler._extract_memory_update(text2, is_memory_channel=False)
    assert payload2 == "Hackathon winners receive $5000 USD."

    # 3. "auto update memory: Mentors are stationed in Lab 3."
    text3 = "auto update memory: Mentors are stationed in Lab 3."
    payload3 = memory_handler._extract_memory_update(text3, is_memory_channel=False)
    assert payload3 == "Mentors are stationed in Lab 3."

    # 4. "remember this: Hardware kits must be returned before 3 PM."
    text4 = "remember this: Hardware kits must be returned before 3 PM."
    payload4 = memory_handler._extract_memory_update(text4, is_memory_channel=False)
    assert payload4 == "Hardware kits must be returned before 3 PM."

    # 5. "@recur add this info in your memory .. Submissions close at 12:00."
    text5 = "@recur add this info in your memory .. Submissions close at 12:00."
    payload5 = memory_handler._extract_memory_update(text5, is_memory_channel=False)
    assert payload5 == "Submissions close at 12:00."

    # 6. Chatter or questions outside memory channel should return None
    assert memory_handler._extract_memory_update("hello how are you?", is_memory_channel=False) is None
    assert memory_handler._extract_memory_update("what is the team limit?", is_memory_channel=False) is None


def test_extract_memory_update_in_memory_channel(memory_handler):
    # Inside #recur-mem-update:
    # Casual chat like "heloow recur" MUST NOT trigger a memory update
    assert memory_handler._extract_memory_update("heloow recur", is_memory_channel=True) is None
    assert memory_handler._extract_memory_update("hello", is_memory_channel=True) is None
    assert memory_handler._extract_memory_update("how are you doing", is_memory_channel=True) is None

    # Normal announcements without explicit trigger MUST NOT trigger a memory update
    announcement = "The Wi-Fi network is RecurGuest and password is hackathon2026."
    assert memory_handler._extract_memory_update(announcement, is_memory_channel=True) is None

    # Questions inside #recur-mem-update MUST NOT trigger a memory update (so they can be answered via Q&A)
    assert memory_handler._extract_memory_update("What is the Wi-Fi password?", is_memory_channel=True) is None
    assert memory_handler._extract_memory_update("Can solo participants submit a project?", is_memory_channel=True) is None
    assert memory_handler._extract_memory_update("When do project submissions close?", is_memory_channel=True) is None

    # ONLY explicit triggers update memory:
    assert memory_handler._extract_memory_update("@recur add this info in your memory .. Wi-Fi password is recur", is_memory_channel=True) == "Wi-Fi password is recur"
    assert memory_handler._extract_memory_update("add to memory: Check-in starts at 9 AM", is_memory_channel=True) == "Check-in starts at 9 AM"
    assert memory_handler._extract_memory_update("auto update memory: Judging at 2 PM", is_memory_channel=True) == "Judging at 2 PM"
    assert memory_handler._extract_memory_update("remember this: Badges required for lunch", is_memory_channel=True) == "Badges required for lunch"


@pytest.mark.asyncio
async def test_handle_memory_update_action(memory_handler, temp_env):
    cfg, db, _ = temp_env

    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"

    channel = MagicMock()
    channel.id = 112233
    channel.name = "recur-mem-update"

    author = MagicMock()
    author.id = 555666
    author.name = "OrganizerSam"
    author.display_name = "Sam (Organizer)"

    message = MagicMock()
    message.author = author
    message.channel = channel
    message.reply = AsyncMock()

    info = "Opening ceremony keynote speaker is Dr. Alan Turing at 10 AM in Main Auditorium."

    success = await memory_handler._handle_memory_update(message, info, bot_user)
    assert success is True

    # 1. Check knowledge/memory_updates.md was created and contains the content
    kb_file = cfg.knowledge_dir / "memory_updates.md"
    assert kb_file.exists()
    file_content = kb_file.read_text(encoding="utf-8")
    assert "Opening ceremony keynote speaker is Dr. Alan Turing" in file_content
    assert "Sam (Organizer)" in file_content

    # 2. Check Database record
    records = db.get_memory_updates(limit=10)
    assert len(records) == 1
    assert records[0]["content"] == info
    assert records[0]["author_name"] == "Sam (Organizer)"
    assert str(records[0]["user_id"]) == "555666"

    # 3. Check FAISS indexer and retriever were invoked
    memory_handler.indexer.build_index.assert_called_once()
    memory_handler.retriever.load.assert_called_once()

    # 4. Check reply sent to Discord channel
    message.reply.assert_called_once()
    call_args = message.reply.call_args[0][0]
    assert "Memory Updated Successfully" in call_args
    assert "knowledge/memory_updates.md" in call_args


@pytest.mark.asyncio
async def test_handle_message_in_memory_channel_allows_admin(memory_handler):
    """Verifies that Admin / Staff members are NOT ignored in #recur-mem-update."""
    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"
    bot_user.bot = True

    channel = MagicMock()
    channel.id = 888888
    channel.name = "recur-mem-update"
    channel.parent = None

    # Admin user
    admin_user = MagicMock()
    admin_user.id = 101010
    admin_user.name = "LeadAdmin"
    admin_user.display_name = "LeadAdmin"
    admin_user.bot = False
    admin_role = MagicMock()
    admin_role.name = "Admin"
    admin_user.roles = [admin_role]

    msg = MagicMock()
    msg.id = 777111
    msg.author = admin_user
    msg.channel = channel
    msg.guild = MagicMock()
    msg.guild.me = MagicMock()
    msg.guild.me.roles = []
    msg.mentions = [bot_user]
    msg.reference = None
    msg.content = "<@999999> add this info in your memory .. Judging rubric: 40% tech, 30% design, 30% pitch."
    msg.reply = AsyncMock()

    await memory_handler.handle_message(msg, bot_user)

    # Bot should reply confirming memory update, even though author is Admin!
    msg.reply.assert_called_once()
    reply_text = msg.reply.call_args[0][0]
    assert "Memory Updated Successfully" in reply_text


@pytest.mark.asyncio
async def test_handle_message_qa_in_memory_channel(memory_handler):
    """Verifies that any user or staff can ask a question in #recur-mem-update and receive an answer."""
    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"
    bot_user.bot = True

    channel = MagicMock()
    channel.id = 888888
    channel.name = "recur-mem-update"
    channel.parent = None

    # Core Member asking a question to test memory
    staff_user = MagicMock()
    staff_user.id = 202020
    staff_user.name = "CoreOrganizer"
    staff_user.display_name = "CoreOrganizer"
    staff_user.bot = False
    core_role = MagicMock()
    core_role.name = "Core Member"
    staff_user.roles = [core_role]

    msg = MagicMock()
    msg.id = 777222
    msg.author = staff_user
    msg.channel = channel
    msg.guild = MagicMock()
    msg.guild.me = MagicMock()
    msg.guild.me.roles = []
    msg.mentions = []
    msg.reference = None
    msg.content = "What time is dinner served tonight?"
    msg.reply = AsyncMock()

    await memory_handler.handle_message(msg, bot_user)

    # Bot should answer via generator!
    memory_handler.generator.generate_answer.assert_called_once()
    msg.reply.assert_called_once()


@pytest.mark.asyncio
async def test_heloow_recur_treated_as_normal_chat_not_memory_update(memory_handler, temp_env):
    """Verifies that 'heloow recur' in #recur-mem-update does NOT trigger a memory update."""
    _, db, _ = temp_env

    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"
    bot_user.bot = True

    channel = MagicMock()
    channel.id = 888888
    channel.name = "recur-mem-update"
    channel.parent = None

    admin_user = MagicMock()
    admin_user.id = 101010
    admin_user.name = "aryarakshit"
    admin_user.display_name = "aryarakshit"
    admin_user.bot = False
    admin_role = MagicMock()
    admin_role.name = "Admin"
    admin_user.roles = [admin_role]

    msg = MagicMock()
    msg.id = 777333
    msg.author = admin_user
    msg.channel = channel
    msg.guild = MagicMock()
    msg.guild.me = MagicMock()
    msg.guild.me.roles = []
    msg.mentions = []
    msg.reference = None
    msg.content = "heloow recur"
    msg.reply = AsyncMock()

    await memory_handler.handle_message(msg, bot_user)

    # 1. Verify NO memory update was logged in database
    records = db.get_memory_updates()
    assert len(records) == 0, "No memory updates should be recorded for 'heloow recur'"

    # 2. Verify bot replied as normal chat (via generator), NOT "Memory Updated Successfully"
    msg.reply.assert_called_once()
    reply_text = msg.reply.call_args[0][0]
    assert "Memory Updated Successfully" not in reply_text


@pytest.mark.asyncio
async def test_catch_up_unanswered_messages_in_memory_channel(memory_handler):
    """Verifies catch_up processes missed memory updates in #recur-mem-update."""
    bot_user = MagicMock()
    bot_user.id = 999999
    bot_user.name = "Recur"
    bot_user.bot = True

    mem_channel = MagicMock()
    mem_channel.id = 888888
    mem_channel.name = "recur-mem-update"
    mem_channel.parent = None

    guild = MagicMock()
    guild.text_channels = [mem_channel]

    admin_user = MagicMock()
    admin_user.id = 303030
    admin_user.name = "SuperAdmin"
    admin_user.bot = False
    admin_role = MagicMock()
    admin_role.name = "Admin"
    admin_user.roles = [admin_role]

    missed_msg = MagicMock()
    missed_msg.id = 991122
    missed_msg.author = admin_user
    missed_msg.channel = mem_channel
    missed_msg.guild = guild
    missed_msg.mentions = []
    missed_msg.reference = None
    missed_msg.content = "add this info in your memory .. Midnight pizza arriving at 12:30 AM."
    missed_msg.reply = AsyncMock()

    async def async_history(limit=20):
        yield missed_msg

    mem_channel.history = async_history

    count = await memory_handler.catch_up_unanswered_messages(bot_user, [guild])
    assert count == 1
    missed_msg.reply.assert_called_once()
    assert "Memory Updated Successfully" in missed_msg.reply.call_args[0][0]


def test_extract_chained_trigger_payload(memory_handler):
    """Verifies that chained trigger phrases like 'update memory/ remember this:' extract clean information."""
    text = '@Recur update memory/ remember this: if anyone asked for "Prize pool" say "not yet disclosed".'
    payload = memory_handler._extract_memory_update(text, is_memory_channel=True)
    assert payload == 'if anyone asked for "Prize pool" say "not yet disclosed".'


def test_hybrid_retrieval_finds_memory_update(memory_handler):
    """Verifies that 'so recur what is the pricepool?' correctly retrieves the organizer's memory update."""
    from rag.retriever import KnowledgeRetriever
    from rag.models import DocumentChunk
    from ai.embeddings import LocalEmbeddingProvider

    emb = LocalEmbeddingProvider()
    mem_chunk = DocumentChunk(
        chunk_id="memory_updates.md#0",
        source="memory_updates.md",
        section="Memory Update by aryarakshit (2026-09-21 02:04:00 UTC)",
        updated_at="2026-09-21",
        text='''Document: Recur Dynamic Memory & Live Organizer Updates (memory_updates.md)
Section: Memory Update by aryarakshit (2026-09-21 02:04:00 UTC)

- Channel: #recur-mem-update
- Author: aryarakshit (101010)
- Information:
  if anyone asked for "Prize pool" say "not yet disclosed".''',
        metadata={"doc_title": "Recur Dynamic Memory & Live Organizer Updates"},
    )
    retriever = KnowledgeRetriever(memory_handler.config.faiss_index_path, memory_handler.config.metadata_path, emb)
    retriever.chunks.append(mem_chunk)

    results = retriever.retrieve("so recur what is the pricepool?")
    assert len(results) > 0
    # Top result MUST be the memory update!
    assert results[0].chunk.source == "memory_updates.md"
    assert "not yet disclosed" in results[0].chunk.text
