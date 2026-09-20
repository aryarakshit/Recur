"""Discord message handler implementing the two-stage decision pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, TYPE_CHECKING
import discord

if TYPE_CHECKING:
    from ai.classifier import MessageClassifier
    from ai.generator import AnswerGenerator
    from config import Config
    from rag.indexer import KnowledgeIndexer
    from rag.retriever import KnowledgeRetriever
    from storage.database import Database
    from storage.memory import ConversationMemory

logger = logging.getLogger(__name__)


class SafeTyping:
    def __init__(self, channel: discord.abc.Messageable) -> None:
        self.channel = channel
        self.ctx = None

    async def __aenter__(self) -> SafeTyping:
        try:
            self.ctx = self.channel.typing()
            await self.ctx.__aenter__()
        except Exception as e:
            logger.debug("Typing indicator suppressed in #%s: %s", getattr(self.channel, "name", "channel"), e)
            self.ctx = None
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.ctx:
            try:
                await self.ctx.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


class MessageHandler:
    def __init__(
        self,
        classifier: MessageClassifier,
        generator: AnswerGenerator,
        retriever: KnowledgeRetriever,
        memory: ConversationMemory,
        database: Database,
        config: Config,
        indexer: KnowledgeIndexer | None = None,
    ) -> None:
        self.classifier = classifier
        self.generator = generator
        self.retriever = retriever
        self.memory = memory
        self.db = database
        self.config = config
        self.indexer = indexer
        self.last_team_ping: dict[int, float] = {}
        self._member_cache: dict[int, tuple[discord.Member, float]] = {}

    async def _resolve_member(
        self,
        author: discord.User | discord.Member,
        guild: discord.Guild | None,
    ) -> discord.Member | None:
        """Resolves a discord.User or discord.Member into a full guild Member with loaded roles."""
        if hasattr(author, "roles") and author.roles:
            return author

        if not guild:
            return None

        # Check in-memory cache (TTL 300s)
        cached = self._member_cache.get(author.id)
        if cached:
            mem, ts = cached
            if time.time() - ts < 300.0:
                return mem

        # Try guild local cache
        get_mem = getattr(guild, "get_member", None)
        if callable(get_mem):
            try:
                mem = get_mem(author.id)
                if mem and hasattr(mem, "roles") and mem.roles:
                    self._member_cache[author.id] = (mem, time.time())
                    return mem
            except Exception:
                pass

        # Fetch from Discord API over HTTP if not cached
        fetch_mem = getattr(guild, "fetch_member", None)
        if callable(fetch_mem):
            try:
                mem = await fetch_mem(author.id)
                if mem and hasattr(mem, "roles"):
                    self._member_cache[author.id] = (mem, time.time())
                    return mem
            except Exception as e:
                logger.debug("Could not fetch member %s from guild: %s", author.id, e)

        return None

    def _clean_content(self, message: discord.Message, bot_user: discord.ClientUser) -> str:
        """Removes bot mention tags from message content to yield the pure question."""
        content = message.content
        # Remove <@id> or <@!id>
        content = re.sub(rf"<@!?{bot_user.id}>", "", content)
        # Strip leading conversational bot address like "@Recur ", "so recur ", "recur "
        content = re.sub(r"^(?:(?:so|hey|hi|yo|ok|okay)\s+)?(?:@?recur\s*[,:]?\s*)", "", content, flags=re.IGNORECASE)
        return content.strip()

    def _is_memory_update_channel(self, channel: Any) -> bool:
        """Verifies if the message was sent in the dedicated memory update channel (e.g. #recur-mem-update)."""
        if not channel:
            return False
        ch_id = getattr(channel, "id", None)
        if self.config.memory_channel_id and ch_id == self.config.memory_channel_id:
            return True

        mem_channels = getattr(
            self.config,
            "memory_channel_names",
            ["recur-mem-update", "recur-mem-updates", "recur-memory", "mem-update", "memory-update", "recur-update"],
        )
        raw_names = []
        name = getattr(channel, "name", None)
        if isinstance(name, str):
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent:
            pname = getattr(parent, "name", None)
            if isinstance(pname, str):
                raw_names.append(pname.lower())

        for raw in raw_names:
            clean = re.sub(r"[^a-z0-9\-]", "", raw).strip("-")
            for mc in mem_channels:
                clean_mc = re.sub(r"[^a-z0-9\-]", "", mc.lower()).strip("-")
                if clean == clean_mc or clean_mc in clean or clean in clean_mc:
                    return True
        return False

    def _is_channel_allowed(self, channel: discord.abc.Messageable) -> bool:
        """Verifies if the message was sent in an allowed channel (e.g. #general, #ask-mentors, or #recur-mem-update)."""
        if self._is_memory_update_channel(channel):
            return True

        allowed = self.config.allowed_channel_names
        if not allowed:
            return True

        raw_names = []
        name = getattr(channel, "name", None)
        if isinstance(name, str):
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent:
            pname = getattr(parent, "name", None)
            if isinstance(pname, str):
                raw_names.append(pname.lower())

        for raw in raw_names:
            clean = re.sub(r"[^a-z0-9\-]", "", raw).strip("-")
            for al in allowed:
                clean_al = re.sub(r"[^a-z0-9\-]", "", al.lower()).strip("-")
                if clean == clean_al or clean_al in clean:
                    return True
        return False

    def _is_team_finding_channel(self, channel: Any) -> bool:
        """Verifies if the message was sent in a team-finding channel (e.g. #find-your-team!)."""
        team_channels = getattr(self.config, "team_finding_channel_names", ["find-your-team", "find-your-team!"])
        if not team_channels:
            team_channels = ["find-your-team", "find-your-team!"]

        raw_names = []
        name = getattr(channel, "name", None)
        if isinstance(name, str):
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent:
            pname = getattr(parent, "name", None)
            if isinstance(pname, str):
                raw_names.append(pname.lower())

        for raw in raw_names:
            clean = re.sub(r"[^a-z0-9\-]", "", raw).strip("-")
            for tc in team_channels:
                clean_tc = re.sub(r"[^a-z0-9\-]", "", tc.lower()).strip("-")
                if clean == clean_tc or clean_tc in clean or clean in clean_tc:
                    return True
        return False

    async def _handle_team_finding_message(
        self,
        message: discord.Message,
        bot_user: discord.ClientUser,
        is_mentioned: bool = False,
    ) -> bool:
        """Handles messages sent in #find-your-team! by pinging @everyone if looking for members.

        Returns:
            True if handled (either replied with @everyone or ignored as non-question chatter).
            False if it is a direct mention question that should fall back to the Q&A pipeline.
        """
        # Ignore bots and Dyno
        if getattr(message.author, "bot", False) or message.author.id == bot_user.id:
            return True
        if "dyno" in getattr(message.author, "name", "").lower():
            return True

        text = self._clean_content(message, bot_user)
        if not text:
            return True

        clean = text.strip().lower()

        # Ignore chatter / noise / greetings
        if self.classifier.is_chatter(clean):
            return True

        # Check if the message is looking for members / team recruitment
        is_teammate = self.classifier.is_teammate_search(clean)

        # In a dedicated team-finding channel, also match non-chatter messages discussing teams or recruitment
        if not is_teammate:
            has_team_word = any(
                w in clean
                for w in ["team", "teams", "teammate", "teammates", "teamate", "teamates", "member", "members", "group", "squad"]
            )
            has_recruitment_word = any(
                w in clean
                for w in [
                    "find", "finding", "seek", "seeking", "search", "searching", "look", "looking",
                    "need", "require", "join", "vacancy", "vacancies",
                    "spot", "spots", "slot", "slots", "open", "available", "dm", "pm",
                    "frontend", "backend", "fullstack", "dev", "developer", "designer", "ai", "ml"
                ]
            )
            # Must not be an official hackathon rule inquiry
            is_rule_query = any(clean.startswith(q) for q in ["what", "can", "is", "are", "how", "where"]) and any(
                term in clean for term in ["limit", "maximum", "rule", "rules", "allowed", "allow", "size", "solo", "minimum"]
            )
            if has_team_word and has_recruitment_word and not is_rule_query and len(clean.split()) >= 3:
                is_teammate = True

        if not is_teammate:
            # If user explicitly asked the bot a question, allow falling through to Q&A
            if is_mentioned:
                return False
            logger.info("Ignoring non-team-search message in team channel: '%s'", text)
            return True

        # Cooldown check: prevent rapid @everyone spam from the same author (60s)
        user_id = message.author.id
        now = time.time()
        last_ping = self.last_team_ping.get(user_id, 0.0)
        if now - last_ping < 60.0:
            logger.info("Skipping @everyone ping for %s in #%s (user cooldown active)", message.author, getattr(message.channel, "name", "channel"))
            return True

        self.last_team_ping[user_id] = now
        logger.info("Triggered team recruitment @everyone reply for %s in #%s", message.author, getattr(message.channel, "name", "channel"))

        reply_content = (
            "@everyone 📢 **Looking for Team Members!**\n"
            "Check out this request above 👆 — reply or DM if you want to team up! 🤝"
        )

        try:
            await message.reply(
                reply_content,
                allowed_mentions=discord.AllowedMentions(everyone=True, replied_user=True),
            )
        except discord.Forbidden:
            logger.warning(
                "Bot lacks 'Mention @everyone' permission in #%s. Falling back to sending without mention.",
                getattr(message.channel, "name", "channel"),
            )
            try:
                await message.reply(
                    reply_content,
                    allowed_mentions=discord.AllowedMentions(everyone=False, replied_user=True),
                )
            except Exception as e:
                logger.error("Failed to send teammate recruitment fallback reply: %s", e)
        except Exception as e:
            logger.error("Failed to send teammate recruitment reply: %s", e)

        return True

    def _get_team_finding_channel(self, guild: discord.Guild | None) -> discord.TextChannel | None:
        """Finds the #find-your-team! text channel in the guild."""
        if not guild:
            return None

        # Check configured channel ID first if available
        team_channel_id = getattr(self.config, "team_finding_channel_id", None)
        if team_channel_id:
            ch = guild.get_channel(team_channel_id)
            if isinstance(ch, discord.TextChannel):
                return ch

        target_names = getattr(self.config, "team_finding_channel_names", ["find-your-team", "find-your-team!"])
        if not target_names:
            target_names = ["find-your-team", "find-your-team!"]

        text_channels = getattr(guild, "text_channels", [])
        for channel in text_channels:
            raw = getattr(channel, "name", "").lower()
            clean = re.sub(r"[^a-z0-9\-]", "", raw).strip("-")
            for target in target_names:
                clean_target = re.sub(r"[^a-z0-9\-]", "", target.lower()).strip("-")
                if clean == clean_target or clean_target in clean or clean in clean_target:
                    return channel
        return None

    async def _forward_team_finding_message(self, message: discord.Message, clean_text: str) -> None:
        """Forwards a teammate recruitment message from #general or #ask-mentors to #find-your-team!"""
        # Cooldown check: prevent rapid @everyone spam from the same author (60s)
        user_id = message.author.id
        now = time.time()
        last_ping = self.last_team_ping.get(user_id, 0.0)
        if now - last_ping < 60.0:
            logger.info("Skipping team recruitment forwarding for %s (user cooldown active)", message.author)
            return

        team_channel = self._get_team_finding_channel(message.guild)
        if not team_channel:
            logger.warning(
                "Could not find team finding channel in guild '%s' to forward message from %s",
                getattr(message.guild, "name", "None"),
                message.author,
            )
            return

        self.last_team_ping[user_id] = now
        logger.info(
            "Forwarding teammate request from %s in #%s to #%s",
            message.author,
            getattr(message.channel, "name", "channel"),
            team_channel.name,
        )

        # Format blockquote for clean Discord embed/display
        quoted_body = "\n".join(f"> {line}" for line in clean_text.splitlines() if line.strip())
        jump_url = getattr(message, "jump_url", "")
        jump_link_str = f"\n🔗 [Jump to Original Message]({jump_url})\n" if jump_url else "\n"
        channel_mention = getattr(message.channel, "mention", f"#{getattr(message.channel, 'name', 'general')}")

        forward_content = (
            f"@everyone 📢 **Looking for Team Members!**\n"
            f"**Participant**: {message.author.mention} (from {channel_mention})\n\n"
            f"{quoted_body}\n"
            f"{jump_link_str}"
            f"Reply or DM {message.author.mention} if you want to team up! 🤝"
        )

        try:
            await team_channel.send(
                forward_content,
                allowed_mentions=discord.AllowedMentions(everyone=True, users=True),
            )
        except discord.Forbidden:
            logger.warning("Bot lacks 'Mention @everyone' permission in #%s", team_channel.name)
            try:
                await team_channel.send(
                    forward_content,
                    allowed_mentions=discord.AllowedMentions(everyone=False, users=True),
                )
            except Exception as e:
                logger.error("Failed to send forward to #%s without @everyone: %s", team_channel.name, e)
        except Exception as e:
            logger.error("Failed to forward teammate recruitment message to #%s: %s", team_channel.name, e)

        # Reply to the user in the original channel (#general or #ask-mentors)
        source_ch_name = getattr(message.channel, "name", "chat")
        reply_content = (
            f"Hey {message.author.mention}, I've forwarded your teammate request to {team_channel.mention} with an @everyone notification! 🤝\n"
            f"*(Please post future team recruitment messages in {team_channel.mention} to keep #{source_ch_name} focused on hackathon questions!)*"
        )
        try:
            await message.reply(
                reply_content,
                allowed_mentions=discord.AllowedMentions(everyone=False, replied_user=True),
            )
        except Exception as e:
            logger.error("Failed to reply to author in #%s: %s", getattr(message.channel, "name", "channel"), e)

    def _extract_memory_update(self, text: str, is_memory_channel: bool = False) -> str | None:
        """Detects if the message is requesting to add/update information in memory.

        ONLY triggers on:
        1. @recur add this info in your memory .. <info>
        2. add to memory: <info>
        3. auto update memory: <info>
        4. remember this: <info>

        Otherwise returns None (normal chat).
        """
        clean = text.strip()
        if not clean:
            return None

        def _clean_payload(info_candidate: str) -> str:
            clean_p = re.sub(r"^[\s.\-–—:/]+", "", info_candidate).strip()
            clean_p = re.sub(
                r"^(?:(?:and\s+)?(?:also\s+)?(?:remember\s+this|remember\s+that|add\s+to\s+memory|add\s+this|update\s+memory|note)\s*[:\-–—./]*\s*)+",
                "",
                clean_p,
                flags=re.IGNORECASE,
            ).strip()
            return clean_p

        # Explicit trigger patterns:
        # 1. "@recur add this info in your memory .. <info>" (or without @recur, or with "to")
        # 2. "add to memory: <info>" (or "add to memory .. <info>")
        # 3. "auto update memory: <info>" (or "auto update memory .. <info>", or "update memory: <info>")
        # 4. "remember this: <info>" (or "remember this .. <info>")
        patterns = [
            r"^(?:@?recur\s+)?(?:please\s+)?add\s+(?:this\s+)?(?:info|information)?\s*(?:in|to|into)\s*(?:your\s+)?memory\s*[:\-–—.]*\s*(.+)$",
            r"^(?:@?recur\s+)?(?:please\s+)?add\s+to\s+memory\s*[:\-–—.]*\s*(.+)$",
            r"^(?:@?recur\s+)?(?:please\s+)?(?:auto\s+)?update\s+memory\s*[:\-–—.]*\s*(.+)$",
            r"^(?:@?recur\s+)?(?:please\s+)?remember\s+this\s*[:\-–—.]*\s*(.+)$",
        ]

        for pat in patterns:
            m = re.match(pat, clean, re.IGNORECASE | re.DOTALL)
            if m:
                info = _clean_payload(m.group(1))
                if len(info) >= 3:
                    return info

        # In-line trigger search (e.g. if user pinged bot in the middle or formatted text):
        trigger_match = re.search(
            r"(?:add\s+(?:this\s+)?(?:info\s+)?(?:in|to)\s+(?:your\s+)?memory|add\s+to\s+memory|(?:auto\s+)?update\s+memory|remember\s+this)\s*[:\-–—.]*\s*(.+)$",
            clean,
            re.IGNORECASE | re.DOTALL,
        )
        if trigger_match:
            info = _clean_payload(trigger_match.group(1))
            if len(info) >= 3:
                return info

        # Otherwise normal chat!
        return None

    async def _handle_memory_update(
        self,
        message: discord.Message,
        info_text: str,
        bot_user: discord.ClientUser,
    ) -> bool:
        """Appends new information to knowledge/memory_updates.md, rebuilds the vector index,
        reloads the retriever, logs to database, and confirms the update to the channel.
        """
        author_name = getattr(message.author, "display_name", getattr(message.author, "name", "Organizer"))
        now_dt = datetime.now(timezone.utc)
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Append to knowledge/memory_updates.md
        kb_file = self.config.knowledge_dir / "memory_updates.md"
        is_new_file = not kb_file.exists()

        entry_text = (
            f"\n\n## Memory Update by {author_name} ({timestamp_str})\n"
            f"- **Channel**: #{getattr(message.channel, 'name', 'recur-mem-update')}\n"
            f"- **Author**: {author_name} ({message.author.id})\n"
            f"- **Information**:\n"
            f"  {info_text.strip()}\n"
        )

        try:
            if is_new_file:
                header = (
                    "# Recur Dynamic Memory & Live Organizer Updates\n"
                    f"*Last Updated: {timestamp_str}*\n\n"
                    "> Official dynamic updates, announcements, and memory additions provided by organizers via #recur-mem-update.\n"
                )
                kb_file.write_text(header + entry_text.strip(), encoding="utf-8")
            else:
                with open(kb_file, "a", encoding="utf-8") as f:
                    f.write(entry_text)
            logger.info("Saved memory update to %s: '%s'", kb_file, info_text[:60])
        except Exception as e:
            logger.error("Failed to write to %s: %s", kb_file, e)
            await message.reply(f"❌ Failed to write memory update to disk: {e}")
            return False

        # 2. Log to database
        try:
            self.db.log_memory_update(
                content=info_text.strip(),
                channel_id=message.channel.id,
                user_id=message.author.id,
                author_name=author_name,
                timestamp=now_dt.timestamp(),
            )
        except Exception as e:
            logger.warning("Could not log memory update to database: %s", e)

        # 3. Auto update memory: Re-index FAISS and reload KnowledgeRetriever
        reindex_details = ""
        try:
            if self.indexer:
                stats = self.indexer.build_index()
                chunk_count = stats.get("chunk_count", 0)
                reindex_details = f"Re-indexed {chunk_count} chunks in {stats.get('time_taken', 0.0)}s."
            if self.retriever:
                self.retriever.load()
            logger.info("Auto-updated memory and re-indexed FAISS successfully.")
        except Exception as e:
            logger.error("Failed to rebuild FAISS index during memory update: %s", e)
            reindex_details = f"Re-indexing notice: {e}"

        # 4. Acknowledge and confirm in Discord
        reply_content = (
            f"🧠 **Memory Updated Successfully!**\n\n"
            f"**Stored Information**:\n"
            f"> {info_text.strip()}\n\n"
            f"✅ **Actions Completed**:\n"
            f"• Written to persistent knowledge base (`knowledge/memory_updates.md`)\n"
            f"• Auto-updated vector index & reloaded retriever ({reindex_details or 'Ready'})\n"
            f"• All future participant queries across `#general`, `#ask-mentors`, and `/ask` will now use this memory! 🚀"
        )

        try:
            await message.reply(
                reply_content,
                allowed_mentions=discord.AllowedMentions(replied_user=True),
            )
            return True
        except Exception as e:
            logger.error("Failed to send memory update confirmation reply: %s", e)
            return False

    def _is_staff_or_bot(
        self,
        author: discord.User | discord.Member,
        bot_user: discord.ClientUser,
        resolved_member: discord.Member | None = None,
    ) -> bool:
        """Checks if the author is a bot, Dyno, Admin, Moderator, Core Member, Volunteer, or Judge."""
        if getattr(author, "bot", False) or author.id == bot_user.id:
            return True
        if "dyno" in getattr(author, "name", "").lower():
            return True

        member = resolved_member or (author if hasattr(author, "roles") else None)
        if member is not None and hasattr(member, "roles"):
            role_names = [getattr(r, "name", "").lower() for r in member.roles]
            perms = getattr(member, "guild_permissions", None)
            is_admin = bool(perms and getattr(perms, "administrator", False)) or any(
                "admin" in r or "administrator" in r for r in role_names
            )
            if is_admin:
                return True
            is_moderator = any("moderator" in r or "mod" in r for r in role_names)
            if is_moderator:
                return True
            for ex in self.config.excluded_role_names:
                if any(ex in r for r in role_names):
                    return True
        return False

    def _is_author_allowed(
        self,
        author: discord.User | discord.Member,
        bot_user: discord.ClientUser,
        is_direct_mention: bool = False,
        resolved_member: discord.Member | None = None,
        channel: Any | None = None,
    ) -> tuple[bool, str]:
        """Checks if the message author is permitted to receive AI answers."""
        if channel and self._is_memory_update_channel(channel):
            if getattr(author, "bot", False) or author.id == bot_user.id:
                return False, "Author is a bot"
            return True, "All users permitted in #recur-mem-update"

        if self._is_staff_or_bot(author, bot_user, resolved_member=resolved_member):
            return False, "Author is staff or a bot"

        member = resolved_member or (author if hasattr(author, "roles") else None)

        if member is not None and hasattr(member, "roles"):
            role_names = [getattr(r, "name", "").lower() for r in member.roles]

            # Check for required 'Hacker' / 'Participant' role (Green)
            if self.config.allowed_role_names:
                has_allowed = any(
                    any(al in r for al in self.config.allowed_role_names)
                    for r in role_names
                )
                if not has_allowed:
                    return False, f"Author does not have required 'Hacker' role (roles: {role_names})"

            return True, "Author is a participant (Hacker)"

        # If author has no guild member roles loaded and could not be verified
        if is_direct_mention:
            return True, "Direct mention from unverified author"

        return False, "Could not verify author has 'Hacker' role from guild roles"

    async def _is_situation_already_resolved(
        self,
        message: discord.Message,
        subsequent_messages: list[discord.Message],
        bot_user: discord.ClientUser,
    ) -> tuple[bool, str]:
        """Inspects subsequent messages in the channel to see if the question was already answered or handled.

        Checks:
        1. Discord native inline reply to this message (by anyone).
        2. Direct mention of the author (via @mention, <@id>, or @name).
        3. Author self-resolution or acknowledgment (e.g. "thanks", "got it", "nvm").
        4. Staff / Mentor response sent in the channel after the question.
        """
        if not subsequent_messages:
            return False, "No subsequent messages"

        author_id = message.author.id
        author_name = getattr(message.author, "name", "").lower()
        author_display = getattr(message.author, "display_name", "").lower()

        def _get_ts(m: discord.Message) -> float:
            cat = getattr(m, "created_at", None)
            if cat is None:
                return 0.0
            if hasattr(cat, "timestamp"):
                return cat.timestamp()
            if isinstance(cat, (int, float)):
                return float(cat)
            return 0.0

        msg_ts = _get_ts(message)

        for sm in subsequent_messages:
            # 1. Check if this subsequent message is an explicit reply to message
            if sm.reference and sm.reference.message_id == message.id:
                author_tag = getattr(sm.author, "name", "User")
                return True, f"Already replied to by {author_tag} (via Discord reply)"

            # Skip bot's own messages for subsequent checks
            if sm.author.id == bot_user.id or getattr(sm.author, "bot", False):
                continue

            # 2. Check author self-resolution: did the question author post saying thanks / got it?
            if sm.author.id == author_id:
                clean_sm = getattr(sm, "content", "").lower().strip()
                if any(re.search(rf"\b{term}\b", clean_sm) for term in [
                    "thanks", "thank you", "thx", "ty", "tysm", "got it", "understood",
                    "nvm", "nevermind", "all clear", "solved", "resolved", "clear now"
                ]):
                    return True, f"Author acknowledged resolution ('{clean_sm[:40]}')"

            # 3. Check if subsequent message explicitly mentions the question author
            if message.author in getattr(sm, "mentions", []):
                return True, f"Addressed by {sm.author.name} (author mentioned)"
            sm_content = getattr(sm, "content", "")
            if f"<@{author_id}>" in sm_content or f"<@!{author_id}>" in sm_content:
                return True, f"Addressed by {sm.author.name} (author ID tagged)"
            if author_name and len(author_name) >= 3 and f"@{author_name}" in sm_content.lower():
                return True, f"Addressed by {sm.author.name} (author @{author_name} tagged)"
            if author_display and len(author_display) >= 3 and f"@{author_display}" in sm_content.lower():
                return True, f"Addressed by {sm.author.name} (author @{author_display} tagged)"

            # 4. Check if a staff member (Admin, Moderator, Core Member, Volunteer, Judge, or Mentor)
            # posted in the channel after the question was asked.
            guild = getattr(message, "guild", None) or getattr(sm, "guild", None)
            resolved_sm_author = await self._resolve_member(sm.author, guild)
            is_staff = self._is_staff_or_bot(sm.author, bot_user, resolved_member=resolved_sm_author)
            if not is_staff and resolved_sm_author and hasattr(resolved_sm_author, "roles"):
                r_names = [getattr(r, "name", "").lower() for r in resolved_sm_author.roles]
                if any("mentor" in r for r in r_names):
                    is_staff = True

            if is_staff:
                sm_ts = _get_ts(sm)
                # If staff member posted within 2 hours after the question (or timestamps zero/mocked):
                if msg_ts == 0.0 or sm_ts == 0.0 or (sm_ts >= msg_ts and sm_ts - msg_ts <= 7200):
                    sm_author_name = getattr(sm.author, "name", "Staff/Mentor")
                    return True, f"Handled by {sm_author_name} in channel after question"

        return False, "No resolution detected"

    async def handle_message(
        self,
        message: discord.Message,
        bot_user: discord.ClientUser,
        situational_context: Optional[str] = None,
    ) -> None:
        """Processes an incoming message and executes the response pipeline."""
        # Always ignore bots and self
        if getattr(message.author, "bot", False) or message.author.id == bot_user.id:
            return
        if "dyno" in getattr(message.author, "name", "").lower():
            return

        # Check if bot is directly mentioned
        is_mentioned = bot_user in message.mentions

        # Check if other users are mentioned (excluding bot)
        other_mentions = [m for m in message.mentions if m.id != bot_user.id]
        has_other_mentions = len(other_mentions) > 0

        # Check if message is a reply to one of the bot's messages or to another user
        is_reply_to_bot = False
        is_reply_to_other = False
        if message.reference and message.reference.message_id:
            try:
                ref_msg = message.reference.resolved
                if not (ref_msg and isinstance(ref_msg, discord.Message)):
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                if ref_msg and isinstance(ref_msg, discord.Message):
                    if ref_msg.author.id == bot_user.id:
                        is_reply_to_bot = True
                    else:
                        is_reply_to_other = True
            except Exception as e:
                logger.debug("Could not resolve referenced message: %s", e)
                is_reply_to_other = not is_mentioned

        # Resolve member with guild roles
        resolved_member = await self._resolve_member(message.author, message.guild)

        # 1. Team Finding Channel Handler (#find-your-team!)
        if self._is_team_finding_channel(message.channel):
            handled = await self._handle_team_finding_message(
                message=message,
                bot_user=bot_user,
                is_mentioned=is_mentioned or is_reply_to_bot,
            )
            if handled:
                return

        cleaned_text = self._clean_content(message, bot_user)
        if not cleaned_text:
            return

        # Check if message is in dedicated memory update channel
        is_mem_channel = self._is_memory_update_channel(message.channel)

        # 1. Check for dynamic memory update requests
        # (e.g. "@recur add this info in your memory ...", or any update in #recur-mem-update)
        memory_payload = self._extract_memory_update(cleaned_text, is_memory_channel=is_mem_channel)
        if memory_payload:
            handled = await self._handle_memory_update(message, memory_payload, bot_user)
            if handled:
                return

        # In ambient chat (not directly mentioned), never reply if message is addressing another user
        if not (is_mentioned or is_reply_to_bot) and not is_mem_channel:
            if has_other_mentions or is_reply_to_other or self.classifier.is_addressed_to_other_user(cleaned_text):
                logger.info("Ignoring message addressed to another user in #%s: '%s'", getattr(message.channel, "name", "channel"), cleaned_text)
                return

            # Check if immediate channel context reveals an already answered question
            if situational_context is None and message.channel and hasattr(message.channel, "history"):
                try:
                    channel_history = [m async for m in message.channel.history(limit=6)]

                    def _get_ts(m: discord.Message) -> float:
                        cat = getattr(m, "created_at", None)
                        if cat and hasattr(cat, "timestamp"):
                            return cat.timestamp()
                        return 0.0

                    msg_ts = _get_ts(message)
                    subsequent = [
                        m for m in channel_history
                        if m.id != message.id and (_get_ts(m) > msg_ts if msg_ts else False)
                    ]
                    if subsequent:
                        resolved, res_reason = await self._is_situation_already_resolved(
                            message=message,
                            subsequent_messages=subsequent,
                            bot_user=bot_user,
                        )
                        if resolved:
                            logger.info(
                                "Ignoring message %s in #%s: %s",
                                message.id,
                                getattr(message.channel, "name", "channel"),
                                res_reason,
                            )
                            return
                except Exception as e:
                    logger.debug("Could not inspect immediate channel history: %s", e)

        # 2. Check if message is a teammate recruitment search in an allowed channel (#general, #ask-mentors, etc.)
        if not is_mem_channel and self.classifier.is_teammate_search(cleaned_text):
            if self._is_channel_allowed(message.channel) or is_mentioned or is_reply_to_bot:
                if not self._is_staff_or_bot(message.author, bot_user, resolved_member=resolved_member):
                    await self._forward_team_finding_message(message, cleaned_text)
                    return

        # 3. Author & Role validation: Only reply to participants, ignore bots & staff (except in #recur-mem-update)
        author_allowed, author_reason = self._is_author_allowed(
            author=message.author,
            bot_user=bot_user,
            is_direct_mention=is_mentioned or is_reply_to_bot,
            resolved_member=resolved_member,
            channel=message.channel,
        )
        if not author_allowed:
            logger.info("Ignoring message from %s: %s", message.author, author_reason)
            return

        # 4. Channel validation: Only reply in 'general' and 'ask-mentors' (unless directly mentioned or in memory update channel)
        if not self._is_channel_allowed(message.channel) and not (is_mentioned or is_reply_to_bot):
            channel_name = getattr(message.channel, "name", "DM")
            logger.info(
                "Ignoring message in non-allowed channel #%s (allowed: %s)",
                channel_name,
                self.config.allowed_channel_names,
            )
            return

        # Get recent channel context for ambiguous classifier decisions
        history_context = self.memory.get_history_context(
            channel_id=message.channel.id,
            user_id=message.author.id,
        )

        decision_context = situational_context or history_context

        # 5. Reply Decision Layer
        if is_mem_channel:
            should_reply, reason = True, "Permitted to reply to any question in #recur-mem-update"
        else:
            should_reply, reason = await self.classifier.should_reply(
                content=cleaned_text,
                is_bot_mentioned=is_mentioned,
                is_reply_to_bot=is_reply_to_bot,
                context=decision_context,
                has_other_mentions=has_other_mentions,
                is_reply_to_other=is_reply_to_other,
            )

        if not should_reply:
            logger.info("Ignoring message '%s': %s", cleaned_text, reason)
            return

        logger.info("Answering message '%s' (Reason: %s)", cleaned_text, reason)

        if message.guild:
            perms = message.channel.permissions_for(message.guild.me)
            bot_roles = [f"'{r.name}'" for r in message.guild.me.roles]
            logger.info(
                "Bot permissions in #%s: send_messages=%s, view_channel=%s, administrator=%s. Bot roles: %s",
                message.channel.name,
                perms.send_messages,
                perms.view_channel,
                perms.administrator,
                ", ".join(bot_roles),
            )

        # Determine organizer channel and tag strings
        organizer_channel_str = self.config.organizer_channel_name
        if self.config.organizer_channel_id:
            organizer_channel_str = f"<#{self.config.organizer_channel_id}>"

        # Resolve Core Member and Volunteer tags dynamically from guild
        tags = []
        if message.guild and hasattr(message.guild, "roles") and isinstance(message.guild.roles, (list, tuple)):
            for r in message.guild.roles:
                r_name = getattr(r, "name", "").lower()
                if r_name in ["core member", "core mem"] and hasattr(r, "mention"):
                    tags.append(r.mention)
                elif r_name in ["volunteer", "voluntear"] and hasattr(r, "mention"):
                    tags.append(r.mention)

        if not tags:
            if self.config.core_member_role_id:
                tags.append(f"<@&{self.config.core_member_role_id}>")
            if self.config.volunteer_role_id:
                tags.append(f"<@&{self.config.volunteer_role_id}>")

        if not tags:
            if self.config.maintainer_role_id:
                tags.append(f"<@&{self.config.maintainer_role_id}>")
            else:
                tags.append(self.config.maintainer_mention)

        organizer_tag_str = " or ".join(tags)

        # 3. Retrieve knowledge and generate answer
        async with SafeTyping(message.channel):
            retrieval_results = self.retriever.retrieve(query=cleaned_text, top_k=4)

            answer, was_fallback = await self.generator.generate_answer(
                question=cleaned_text,
                retrieval_results=retrieval_results,
                history=decision_context or history_context,
                organizer_channel=organizer_channel_str,
                organizer_tag=organizer_tag_str,
            )

            # If answer is fallback, log question for organizers
            if was_fallback:
                sources = [r.chunk.source for r in retrieval_results]
                self.db.log_unanswered_question(
                    question=cleaned_text,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                    retrieved_sources=sources,
                )

                # Ambient suppression: If the bot was NOT directly mentioned or replied to,
                # stay quiet! Do not spam #general or #ask-mentors with robotic fallback messages.
                if not (is_mentioned or is_reply_to_bot) and not is_mem_channel:
                    logger.info(
                        "Suppressed fallback reply in ambient chat (#%s) for '%s' (staying quiet)",
                        getattr(message.channel, "name", "channel"),
                        cleaned_text,
                    )
                    return

            # Send reply
            try:
                await message.reply(answer, mention_author=True)
                logger.info("Successfully replied to %s in #%s", message.author, getattr(message.channel, "name", "channel"))
            except discord.Forbidden as e:
                logger.error("Forbidden: bot lacks Send Messages/View Channel permission in #%s (%s)", getattr(message.channel, "name", "channel"), e)
                try:
                    await message.channel.send(f"{message.author.mention} {answer}")
                except Exception as e2:
                    logger.error("Fallback send also failed in #%s: %s", getattr(message.channel, "name", "channel"), e2)
            except Exception as e:
                logger.error("Failed to send reply to message: %s", e)
                try:
                    await message.channel.send(f"{message.author.mention} {answer}")
                except Exception as e2:
                    logger.error("Fallback send also failed: %s", e2)

            # 4. Update conversation memory
            self.memory.add_user_message(
                channel_id=message.channel.id,
                user_id=message.author.id,
                message=cleaned_text,
            )
            self.memory.add_assistant_message(
                channel_id=message.channel.id,
                user_id=message.author.id,
                message=answer,
            )

    async def catch_up_unanswered_messages(
        self,
        bot_user: discord.ClientUser,
        guilds: list[discord.Guild],
        limit_per_channel: int = 20,
    ) -> int:
        """Scans recent channel history in allowed channels and handles any unanswered participant queries or teammate requests.

        Returns:
            Number of unanswered messages processed.
        """
        processed_count = 0
        for guild in guilds:
            # 1. Identify allowed and team finding channels
            channels_to_check: list[discord.TextChannel] = []
            for ch in getattr(guild, "text_channels", []):
                if self._is_team_finding_channel(ch) or self._is_channel_allowed(ch):
                    channels_to_check.append(ch)

            # 2. Gather forwarded original message IDs in team finding channel to avoid duplicate forwards
            forwarded_msg_ids: set[int] = set()
            team_channel = self._get_team_finding_channel(guild)
            if team_channel:
                try:
                    async for tm in team_channel.history(limit=50):
                        if tm.author.id == bot_user.id:
                            urls = re.findall(r"https://discord\.com/channels/\d+/\d+/(\d+)", tm.content)
                            for u in urls:
                                forwarded_msg_ids.add(int(u))
                except Exception as e:
                    logger.debug("Could not read team channel history: %s", e)

            # 3. Check each channel
            for ch in channels_to_check:
                try:
                    recent_messages: list[discord.Message] = [m async for m in ch.history(limit=limit_per_channel)]
                except Exception as e:
                    logger.warning("Could not fetch history for #%s: %s", getattr(ch, "name", "channel"), e)
                    continue

                if not recent_messages:
                    continue

                # Collect all message IDs that received a reply from the bot
                replied_to_by_bot: set[int] = set()
                for m in recent_messages:
                    if m.author.id == bot_user.id and m.reference and m.reference.message_id:
                        replied_to_by_bot.add(m.reference.message_id)

                # Iterate in chronological order (oldest first)
                chronological_messages = list(reversed(recent_messages))
                for idx, msg in enumerate(chronological_messages):
                    # Ignore bots, Dyno, self
                    if getattr(msg.author, "bot", False) or msg.author.id == bot_user.id:
                        continue
                    if "dyno" in getattr(msg.author, "name", "").lower():
                        continue

                    # If bot already replied directly to this message, skip
                    if msg.id in replied_to_by_bot:
                        continue

                    # Subsequent messages sent in this channel after msg was sent
                    subsequent_messages = chronological_messages[idx + 1 :]

                    # Resolve member with guild roles
                    resolved_member = await self._resolve_member(msg.author, guild)

                    # In dedicated memory update channel, handle missed updates/questions directly!
                    if self._is_memory_update_channel(ch):
                        await self.handle_message(msg, bot_user)
                        replied_to_by_bot.add(msg.id)
                        processed_count += 1
                        continue

                    # Skip bots, Dyno, and staff immediately
                    if self._is_staff_or_bot(msg.author, bot_user, resolved_member=resolved_member):
                        continue

                    cleaned = self._clean_content(msg, bot_user)
                    if not cleaned or self.classifier.is_chatter(cleaned):
                        continue

                    # Check if message is addressed to another user (via mention, reply, or @tag)
                    other_mentions = [m for m in msg.mentions if m.id != bot_user.id]
                    has_other_mentions = len(other_mentions) > 0
                    is_reply_to_other = bool(msg.reference and msg.reference.message_id)
                    if has_other_mentions or is_reply_to_other or self.classifier.is_addressed_to_other_user(cleaned):
                        continue

                    # If message is in team finding channel
                    if self._is_team_finding_channel(ch):
                        if self.classifier.is_teammate_search(cleaned):
                            handled = await self._handle_team_finding_message(msg, bot_user)
                            if handled:
                                replied_to_by_bot.add(msg.id)
                                processed_count += 1
                        continue

                    # If message in allowed channel (#general, #ask-mentors) is a teammate request
                    if self.classifier.is_teammate_search(cleaned):
                        if msg.id in forwarded_msg_ids:
                            continue
                        # Check if teammate search was already resolved (e.g. author got team or someone responded)
                        already_resolved, res_reason = await self._is_situation_already_resolved(
                            message=msg,
                            subsequent_messages=subsequent_messages,
                            bot_user=bot_user,
                        )
                        if already_resolved:
                            logger.info("Catch-up skipping resolved teammate search %s: %s", msg.id, res_reason)
                            continue
                        await self._forward_team_finding_message(msg, cleaned)
                        forwarded_msg_ids.add(msg.id)
                        replied_to_by_bot.add(msg.id)
                        processed_count += 1
                        continue

                    # For ambient Q&A questions, check author role (requires Hacker role)
                    author_allowed, _ = self._is_author_allowed(
                        author=msg.author,
                        bot_user=bot_user,
                        is_direct_mention=bot_user in msg.mentions,
                        resolved_member=resolved_member,
                        channel=ch,
                    )
                    if not author_allowed:
                        continue

                    # Check situational context: Did a mentor/staff member or peer already answer, or did author resolve?
                    already_resolved, res_reason = await self._is_situation_already_resolved(
                        message=msg,
                        subsequent_messages=subsequent_messages,
                        bot_user=bot_user,
                    )
                    if already_resolved:
                        logger.info(
                            "Catch-up skipping message %s from %s in #%s: %s",
                            msg.id,
                            msg.author,
                            getattr(ch, "name", "channel"),
                            res_reason,
                        )
                        continue

                    # Build situational context from subsequent messages if any exist
                    situational_context = None
                    if subsequent_messages:
                        context_lines = []
                        for sm in subsequent_messages[:10]:
                            sm_author = getattr(sm.author, "display_name", getattr(sm.author, "name", "User"))
                            resolved_sm = await self._resolve_member(sm.author, guild)
                            is_sm_staff = self._is_staff_or_bot(sm.author, bot_user, resolved_member=resolved_sm)
                            role_label = "Mentor/Staff" if is_sm_staff else "Participant"
                            context_lines.append(f"[{role_label}] {sm_author}: {sm.content}")
                        situational_context = "Subsequent channel conversation:\n" + "\n".join(context_lines)

                    # Process through cognitive should_reply decision with situational context
                    should_reply, reply_reason = await self.classifier.should_reply(
                        content=cleaned,
                        is_bot_mentioned=bot_user in msg.mentions,
                        is_reply_to_bot=False,
                        context=situational_context,
                        has_other_mentions=has_other_mentions,
                        is_reply_to_other=is_reply_to_other,
                    )
                    if not should_reply:
                        logger.info(
                            "Catch-up skipping message %s from %s in #%s: %s",
                            msg.id,
                            msg.author,
                            getattr(ch, "name", "channel"),
                            reply_reason,
                        )
                        continue

                    await self.handle_message(msg, bot_user, situational_context=situational_context)
                    replied_to_by_bot.add(msg.id)
                    processed_count += 1

        return processed_count
