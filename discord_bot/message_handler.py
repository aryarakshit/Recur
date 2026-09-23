"""Discord message handler implementing the two-stage decision pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, TYPE_CHECKING
import discord

from storage.memory_store import MemoryStore

if TYPE_CHECKING:
    from ai.classifier import MessageClassifier
    from ai.generator import AnswerGenerator
    from config import Config
    from rag.retriever import KnowledgeRetriever
    from storage.database import Database
    from storage.memory import ConversationMemory
    from storage.memory_store import MemoryEntry

logger = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 2000
# Besides #recur-mem-update, these are the only channels Recur answers in.
ANSWER_CHANNELS = {"general", "ask-mentors"}

# Memory commands must start the message ("remember ...", "update mem ...", "delete mem ..."),
# so a sentence that merely mentions a trigger word is never treated as a command.
_LEAD = r"^\s*(?:@?recur\b[\s,:]*)?(?:(?:please|pls|plz|kindly)\s+)?"
# "this" belongs to the trigger only before a delimiter, so "remember this is the venue" keeps "this".
_THIS = r"\s+this(?=\s*(?:[:;.\-–—/>|]|$))"
_ADD_TRIGGERS = (
    r"add\s+(?:this\s+)?(?:info\s+|information\s+)?(?:in|to|into)\s+(?:your\s+)?memory"
    r"|(?:auto\s+)?update\s+memory"
    rf"|remember{_THIS}"
)
_ADD_SHORTHAND = (
    r"(?:auto\s+)?update\s+(?:your\s+)?mem(?:ory)?"
    r"|mem(?:ory)?\s+(?:update|add|save)"
    r"|(?:add|save)\s+(?:this\s+)?(?:to|in|into)\s+mem(?:ory)?"
    r"|save\s+(?:this\s+)?(?:in|to)\s+(?:your\s+)?memory"
    rf"|(?:remember|rmember|remeber|rember)(?:\s+that|{_THIS})?"
    r"|yaad\s+(?:rakhna|rakho)(?:\s+ki)?"
)
_DELETE_TRIGGERS = (
    r"remove\s+(?:this\s+)?(?:info\s+|information\s+)?(?:from|in)\s+(?:your\s+)?mem(?:ory)?"
    r"|remove\s+(?:from\s+)?mem(?:ory)?"
    r"|delete\s+(?:from\s+)?memory"
    r"|forget(?:\s+(?:this|about))?"
)
_DELETE_SHORTHAND = (
    r"(?:delete|del|remove|erase|clear)\s+(?:from\s+)?(?:this\s+|that\s+|the\s+)?mem(?:ory)?"
    r"|(?:delete|del|remove)\s+(?:karo\s+)?mem(?:ory)?"
    r"|(?:delete|del|remove|erase)(?=\s+#?\d+)"
    r"|mem(?:ory)?\s+(?:delete|del|remove)"
)


def _command_re(triggers: str) -> re.Pattern[str]:
    trigger = f"(?:{triggers})"
    # Triggers may be chained, e.g. "remember or update memory: ..." or "update memory/ remember this: ...".
    return re.compile(
        rf"{_LEAD}{trigger}(?:\s*(?:or|and|/|,|&)\s*{trigger})*(?:\b|(?=#))(?P<sep>[\s:;.\-–—/>|]*)(?P<arg>.*)$",
        re.IGNORECASE | re.DOTALL,
    )


_ADD_RE = _command_re(_ADD_TRIGGERS)
_ADD_MEM_CHANNEL_RE = _command_re(f"{_ADD_TRIGGERS}|{_ADD_SHORTHAND}")
_DELETE_RE = _command_re(_DELETE_TRIGGERS)
_DELETE_MEM_CHANNEL_RE = _command_re(f"{_DELETE_TRIGGERS}|{_DELETE_SHORTHAND}")
_LIST_RE = re.compile(
    _LEAD
    + r"(?:(?:list|show|view|see|check)\s+(?:all\s+)?(?:(?:the|your|my|saved)\s+)?mem(?:ory|ories|s)?"
    r"|mem(?:ory|ories)?\s+list|what\s+do\s+you\s+remember)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_ID_LIST_RE = re.compile(r"(?:#?\s*\d+\s*(?:,|&|and)?\s*)+", re.IGNORECASE)
# Arguments that point at the replied-to message instead of carrying text.
_THIS_WORDS = {
    "", "this", "that", "it", "this one", "that one", "this mem", "this memory",
    "that mem", "that memory", "this note", "this info",
}
# Instructions typed without a trigger ("if anyone asks ..., say ...") get a hint instead
# of being saved silently.
_DIRECTIVE_RE = re.compile(
    r"^\s*(?:if\s+(?:any|some)\s*(?:one|body)\b|if\s+(?:a\s+)?(?:participant|hacker|user)s?\b"
    r"|whenever\b|from\s+now\s+on\b|tell\s+(?:them|everyone|participants|people|hackers|users)\b)",
    re.IGNORECASE,
)
_QUOTE_PAIRS = (('"""', '"""'), ("'''", "'''"), ("```", "```"), ('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"))

# Replies never ping @everyone/@here; memory cards echo organizer text, so they ping nobody.
NO_EVERYONE_PING = discord.AllowedMentions(everyone=False)
MEMORY_CARD_MENTIONS = discord.AllowedMentions(everyone=False, users=False, roles=False, replied_user=True)


def _unwrap_quotes(text: str) -> str:
    """Strips quotes that wrap the whole payload, e.g. remember \"\"\"<note>\"\"\"."""
    text = text.strip()
    for open_q, close_q in _QUOTE_PAIRS:
        if len(text) <= len(open_q) + len(close_q):
            continue
        inner = text[len(open_q):-len(close_q)]
        if text.startswith(open_q) and text.endswith(close_q) and open_q not in inner and close_q not in inner:
            return inner.strip()
    return text


def _fit_discord(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


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
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.classifier = classifier
        self.generator = generator
        self.retriever = retriever
        self.memory = memory
        self.db = database
        self.config = config
        self.memory_store = memory_store or MemoryStore(config.knowledge_dir)
        self.last_team_ping: dict[int, float] = {}
        self._member_cache: dict[int, tuple[discord.Member, float]] = {}
        self._user_last_query: dict[int, float] = {}

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
        # Exact names only: a substring match made channels like #updates or #recur count
        # as the memory channel, letting anyone there rewrite the bot's memory.
        return self._channel_matches(channel, set(self.config.memory_channel_names))

    def _is_channel_allowed(self, channel: discord.abc.Messageable) -> bool:
        """Verifies that replies are limited to general, ask-mentors, or memory updates."""
        return self._is_response_channel(channel)

    def _is_response_channel(self, channel: Any) -> bool:
        """Return whether the bot is allowed to process messages in this channel.

        Recur only ever answers in #general, #ask-mentors and #recur-mem-update.
        """
        return self._is_memory_update_channel(channel) or self._channel_matches(channel, ANSWER_CHANNELS)

    @staticmethod
    def _home_channel_name(channel: Any) -> str | None:
        """Name of the channel a message lives in (the parent channel for threads)."""
        for ch in (getattr(channel, "parent", None), channel):
            name = getattr(ch, "name", None)
            if isinstance(name, str):
                return name.lstrip("#")
        return None

    def _channel_matches(self, channel: Any, names: set[str]) -> bool:
        raw_names = []
        name = getattr(channel, "name", None)
        if isinstance(name, str):
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent and isinstance(getattr(parent, "name", None), str):
            raw_names.append(parent.name.lower())
        for raw in raw_names:
            clean = re.sub(r"[^a-z0-9\-]", "", raw).strip("-")
            if clean in {re.sub(r"[^a-z0-9\-]", "", n).strip("-") for n in names}:
                return True
        return False

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

    def _parse_memory_command(self, text: str, is_memory_channel: bool = False) -> tuple[str, str] | None:
        """Parses a memory command at the start of a message.

        Returns (action, argument) with action "add", "delete" or "list", or None for
        normal chat. An empty argument means the command targets the replied-to message
        ("remember this", "delete this mem"). Shorthand triggers such as "remember ...",
        "update mem ..." and "delete mem ..." only apply in #recur-mem-update.
        """
        clean = text.strip()
        if not clean:
            return None
        if is_memory_channel and _LIST_RE.match(clean):
            return "list", ""

        delete_re, add_re = (
            (_DELETE_MEM_CHANNEL_RE, _ADD_MEM_CHANNEL_RE) if is_memory_channel else (_DELETE_RE, _ADD_RE)
        )
        m = delete_re.match(clean)
        if m:
            target = _unwrap_quotes(m.group("arg")).rstrip(".!")
            return "delete", "" if target.lower() in _THIS_WORDS else target

        m = add_re.match(clean)
        if m:
            info = _unwrap_quotes(m.group("arg"))
            if info.lower().rstrip(".!") in _THIS_WORDS:
                return "add", ""
            # "remember when the deadline was?" is a question, not a note to save.
            if info.endswith("?") and ":" not in m.group("sep"):
                return None
            return ("add", info) if len(info) >= 3 else None
        return None

    def _extract_memory_update(self, text: str, is_memory_channel: bool = False) -> str | None:
        """Returns the note to save if the message is a memory-add command with text."""
        command = self._parse_memory_command(text, is_memory_channel)
        return command[1] if command and command[0] == "add" and command[1] else None

    def _extract_memory_removal(self, text: str, is_memory_channel: bool = False) -> str | None:
        """Returns the delete target (ID or keywords) if the message is a memory-delete command."""
        command = self._parse_memory_command(text, is_memory_channel)
        return command[1] if command and command[0] == "delete" and command[1] else None

    async def _run_memory_command(
        self,
        message: discord.Message,
        action: str,
        argument: str,
        referenced: discord.Message | None,
        bot_user: discord.ClientUser,
    ) -> None:
        if action == "list":
            reply = self._format_memory_list()
        elif action == "add":
            reply = self._save_memory(message, argument, referenced)
        else:
            reply = self._delete_memory(argument, referenced, bot_user)
        try:
            await message.reply(_fit_discord(reply), allowed_mentions=MEMORY_CARD_MENTIONS)
        except Exception as e:
            logger.error("Failed to send memory %s reply: %s", action, e)

    def _save_memory(self, message: discord.Message, info: str, referenced: discord.Message | None) -> str:
        # "remember this" as a reply saves the replied-to message.
        text = info or (getattr(referenced, "content", "") or "").strip()
        if not text:
            return "What should I remember? Send `remember <note>`, or reply `remember this` to a message."

        author_name = getattr(message.author, "display_name", getattr(message.author, "name", "Organizer"))
        entry = self.memory_store.add(text, author=author_name, author_id=message.author.id)
        try:
            self.db.log_memory_update(
                content=entry.text,
                channel_id=message.channel.id,
                user_id=message.author.id,
                author_name=author_name,
                timestamp=datetime.now(timezone.utc).timestamp(),
            )
        except Exception as e:
            logger.warning("Could not log memory update to database: %s", e)

        return (
            f"🧠 Saved as memory #{entry.id}\n"
            f"> {entry.preview(1500)}\n"
            f"I'll follow this in #general and #ask-mentors until it's deleted "
            f"(`delete mem #{entry.id}`, or reply `delete this mem`)."
        )

    def _delete_memory(
        self,
        target: str,
        referenced: discord.Message | None,
        bot_user: discord.ClientUser,
    ) -> str:
        if not target:
            ids = self._memory_ids_in_reference(referenced, bot_user)
            if not ids:
                return (
                    "Which memory? Send `delete mem #N` (see `list mem`), "
                    "or reply `delete this mem` to my \"Saved as memory\" message."
                )
        elif _ID_LIST_RE.fullmatch(target):
            ids = {int(n) for n in re.findall(r"\d+", target)}
        else:
            matches = self.memory_store.find(target)
            if not matches:
                return f"🔍 No memory matches `{target}`. Send `list mem` to see what's saved."
            if len(matches) > 1:
                options = "\n".join(f"• #{e.id} — {e.preview(120)}" for e in matches[:10])
                return f"{len(matches)} memories match `{target}`. Which one?\n{options}\nSend `delete mem #N`."
            ids = {matches[0].id}

        removed = self.memory_store.delete(ids)
        if not removed:
            missing = ", ".join(f"#{i}" for i in sorted(ids))
            return f"🔍 No memory {missing}. Send `list mem` to see what's saved."

        for entry in removed:
            try:
                self.db.delete_memory_update(entry.text)
            except Exception as e:
                logger.warning("Could not delete memory #%d from database: %s", entry.id, e)

        lines = "\n".join(f"> #{e.id} — {e.preview(300)}" for e in removed)
        if len(removed) == 1:
            return f"🗑️ Deleted memory #{removed[0].id}\n{lines}\nI won't use it anymore."
        return f"🗑️ Deleted {len(removed)} memories\n{lines}\nI won't use them anymore."

    def _memory_ids_in_reference(
        self,
        referenced: discord.Message | None,
        bot_user: discord.ClientUser,
    ) -> set[int]:
        """Finds which memory a "delete this mem" reply points at."""
        if referenced is None:
            return set()
        content = getattr(referenced, "content", "") or ""
        if getattr(referenced.author, "id", None) == bot_user.id:
            ids = {int(n) for n in re.findall(r"Saved as memory #(\d+)", content)}
            return ids if len(ids) == 1 else set()
        # Reply to the original message that was saved with "remember this".
        flat = " ".join(content.split()).lower()
        if not flat:
            return set()
        return {e.id for e in self.memory_store.list() if " ".join(e.text.split()).lower() == flat}

    def _format_memory_list(self) -> str:
        entries: list[MemoryEntry] = self.memory_store.list()
        if not entries:
            return "🧠 No saved memories yet. Add one with `remember <note>`."
        header = f"🧠 Saved memories ({len(entries)}):"
        footer = "Delete one with `delete mem #N`."
        budget = DISCORD_MESSAGE_LIMIT - len(header) - len(footer) - 40
        lines: list[str] = []
        for i, e in enumerate(entries):
            line = f"• #{e.id} — {e.preview(150)}"
            if sum(len(ln) + 1 for ln in lines) + len(line) > budget:
                lines.append(f"…and {len(entries) - i} more")
                break
            lines.append(line)
        return "\n".join([header, *lines, footer])

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
        situational_context: str | None = None,
    ) -> None:
        """Processes an incoming message and executes the response pipeline."""
        # Always ignore bots and self
        if getattr(message.author, "bot", False) or message.author.id == bot_user.id:
            return
        if "dyno" in getattr(message.author, "name", "").lower():
            return

        # Check if message is in dedicated memory update channel
        is_mem_channel = self._is_memory_update_channel(message.channel)

        # Per-user query throttle (1.5s) under heavy demand to prevent burst spam from exhausting API quotas
        now = time.time()
        user_id = getattr(message.author, "id", 0)
        if not is_mem_channel and user_id:
            last_ts = self._user_last_query.get(user_id, 0.0)
            if now - last_ts < 1.5:
                logger.info("Throttling burst query from %s (last query %.2fs ago)", message.author, now - last_ts)
                return
            self._user_last_query[user_id] = now

        # Periodic memory cleanup for 24x7 operation
        if len(self._user_last_query) > 1000:
            cutoff = now - 300.0
            self._user_last_query = {uid: ts for uid, ts in self._user_last_query.items() if ts > cutoff}
        if len(self._member_cache) > 1000:
            cutoff = now - 300.0
            self._member_cache = {uid: val for uid, val in self._member_cache.items() if val[1] > cutoff}
        if len(self.last_team_ping) > 500:
            cutoff = now - 120.0
            self.last_team_ping = {uid: ts for uid, ts in self.last_team_ping.items() if ts > cutoff}

        # Check if bot is directly mentioned
        is_mentioned = bot_user in message.mentions

        # Check if other users are mentioned (excluding bot)
        other_mentions = [m for m in message.mentions if m.id != bot_user.id]
        has_other_mentions = len(other_mentions) > 0

        if not self._is_response_channel(message.channel):
            logger.info("Ignoring message in non-response channel #%s", getattr(message.channel, "name", "channel"))
            return

        # Check if message is a reply to one of the bot's messages or to another user
        is_reply_to_bot = False
        is_reply_to_other = False
        ref_msg: discord.Message | None = None
        if message.reference and message.reference.message_id:
            try:
                resolved = message.reference.resolved
                if not isinstance(resolved, discord.Message):
                    resolved = await message.channel.fetch_message(message.reference.message_id)
                if isinstance(resolved, discord.Message):
                    ref_msg = resolved
                    if ref_msg.author.id == bot_user.id:
                        is_reply_to_bot = True
                    else:
                        is_reply_to_other = True
            except Exception as e:
                logger.debug("Could not resolve referenced message: %s", e)
                is_reply_to_other = not is_mentioned

        # Resolve member with guild roles
        resolved_member = await self._resolve_member(message.author, message.guild)

        cleaned_text = self._clean_content(message, bot_user)
        if not cleaned_text:
            return

        # Memory commands only run in #recur-mem-update, so participants in #general or
        # #ask-mentors can never change what the bot tells everyone.
        if is_mem_channel:
            command = self._parse_memory_command(cleaned_text, is_memory_channel=True)
            if command:
                await self._run_memory_command(message, command[0], command[1], ref_msg, bot_user)
                return
            if self.classifier.is_chatter(cleaned_text) or has_other_mentions or is_reply_to_other:
                logger.info("Ignoring chatter in #%s: '%s'", getattr(message.channel, "name", "channel"), cleaned_text)
                return
            if _DIRECTIVE_RE.match(cleaned_text) and not cleaned_text.rstrip().endswith("?"):
                try:
                    await message.reply(
                        "Not saved. To save an instruction, start with `remember`, e.g. "
                        "`remember if anyone asks about X, say Y`.",
                        allowed_mentions=MEMORY_CARD_MENTIONS,
                    )
                except Exception as e:
                    logger.error("Failed to send memory hint: %s", e)
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

        # 2. Check if message is a teammate recruitment search in #general or #ask-mentors
        if not is_mem_channel and self.classifier.is_teammate_search(cleaned_text):
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
        start_time = time.time()
        async with SafeTyping(message.channel):
            retrieval_results = self.retriever.retrieve(query=cleaned_text, top_k=4)

            answer, was_fallback = await self.generator.generate_answer(
                question=cleaned_text,
                retrieval_results=retrieval_results,
                history=decision_context or history_context,
                organizer_channel=organizer_channel_str,
                organizer_tag=organizer_tag_str,
                channel_name=self._home_channel_name(message.channel),
            )
            answer = _fit_discord(answer)
            latency = time.time() - start_time
            try:
                self.db.log_query(
                    question=cleaned_text,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                    latency=latency,
                    was_fallback=was_fallback,
                )
            except Exception as e:
                logger.debug("Could not log query execution: %s", e)

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
                await message.reply(answer, mention_author=True, allowed_mentions=NO_EVERYONE_PING)
                logger.info("Successfully replied to %s in #%s", message.author, getattr(message.channel, "name", "channel"))
            except discord.Forbidden as e:
                logger.error("Forbidden: bot lacks Send Messages/View Channel permission in #%s (%s)", getattr(message.channel, "name", "channel"), e)
                try:
                    await message.channel.send(
                        _fit_discord(f"{message.author.mention} {answer}"), allowed_mentions=NO_EVERYONE_PING
                    )
                except Exception as e2:
                    logger.error("Fallback send also failed in #%s: %s", getattr(message.channel, "name", "channel"), e2)
            except Exception as e:
                logger.error("Failed to send reply to message: %s", e)
                try:
                    await message.channel.send(
                        _fit_discord(f"{message.author.mention} {answer}"), allowed_mentions=NO_EVERYONE_PING
                    )
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
            # 1. Only the channels Recur answers in: #general, #ask-mentors, #recur-mem-update
            channels_to_check: list[discord.TextChannel] = [
                ch for ch in getattr(guild, "text_channels", []) if self._is_response_channel(ch)
            ]

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
