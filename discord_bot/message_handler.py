"""Discord message handler implementing the two-stage decision pipeline."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, TYPE_CHECKING
import discord

if TYPE_CHECKING:
    from ai.classifier import MessageClassifier
    from ai.generator import AnswerGenerator
    from config import Config
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
    ) -> None:
        self.classifier = classifier
        self.generator = generator
        self.retriever = retriever
        self.memory = memory
        self.db = database
        self.config = config
        self.last_team_ping: dict[int, float] = {}

    def _clean_content(self, message: discord.Message, bot_user: discord.ClientUser) -> str:
        """Removes bot mention tags from message content to yield the pure question."""
        content = message.content
        # Remove <@id> or <@!id>
        content = re.sub(rf"<@!?{bot_user.id}>", "", content)
        return content.strip()

    def _is_channel_allowed(self, channel: discord.abc.Messageable) -> bool:
        """Verifies if the message was sent in an allowed channel (e.g. #general or #ask-mentors)."""
        allowed = self.config.allowed_channel_names
        if not allowed:
            return True

        raw_names = []
        name = getattr(channel, "name", None)
        if name:
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent and getattr(parent, "name", None):
            raw_names.append(parent.name.lower())

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
        if name:
            raw_names.append(name.lower())
        parent = getattr(channel, "parent", None)
        if parent and getattr(parent, "name", None):
            raw_names.append(parent.name.lower())

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
            has_team_word = any(w in clean for w in ["team", "teammate", "teammates", "member", "members", "group", "squad"])
            has_recruitment_word = any(
                w in clean for w in [
                    "looking", "need", "require", "seeking", "join", "vacancy", "vacancies",
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

    def _is_author_allowed(
        self,
        author: discord.User | discord.Member,
        bot_user: discord.ClientUser,
        is_direct_mention: bool = False,
    ) -> tuple[bool, str]:
        """Checks if the message author is permitted to receive AI answers.

        Rules:
        1. Always ignore bots (author.bot is True, bot_user, or name matching 'dyno').
        2. Never reply to staff / organizers (admin, administrator, moderator, core member, volunteer, judge, bot, dyno)
           or users with administrator permissions.
        3. Only reply to participants with the 'Hacker' / 'Participant' role.
        """
        if getattr(author, "bot", False) or author.id == bot_user.id:
            return False, "Author is a bot"

        author_name = getattr(author, "name", "").lower()
        if "dyno" in author_name:
            return False, "Author is Dyno"

        # If in a guild (discord.Member), inspect roles and administrator permissions
        if hasattr(author, "roles"):
            role_names = [r.name.lower() for r in author.roles]
            perms = getattr(author, "guild_permissions", None)
            is_admin = bool(perms and getattr(perms, "administrator", False)) or any(
                "admin" in r or "administrator" in r for r in role_names
            )
            if is_admin:
                return False, "Author has Administrator permissions or Admin role"

            is_moderator = any("moderator" in r or "mod" in r for r in role_names)
            if is_moderator:
                return False, "Author has Moderator role"

            # Check for excluded staff roles (core member, volunteer, judge, bot, dyno)
            for ex in self.config.excluded_role_names:
                if any(ex in r for r in role_names):
                    return False, f"Author has excluded staff role '{ex}'"

            # Check for required 'Hacker' / 'Participant' role
            if self.config.allowed_role_names:
                has_allowed = any(
                    any(al in r for al in self.config.allowed_role_names)
                    for r in role_names
                )
                if not has_allowed:
                    return False, f"Author does not have required 'Hacker' role (roles: {role_names})"

        # Explicit @Recur mentions for verified participants are allowed
        if is_direct_mention:
            return True, "Direct mention to bot"

        return True, "Author is a participant (Hacker)"

    async def handle_message(self, message: discord.Message, bot_user: discord.ClientUser) -> None:
        """Processes an incoming message and executes the response pipeline."""
        # Always ignore bots and self
        if getattr(message.author, "bot", False) or message.author.id == bot_user.id:
            return
        if "dyno" in getattr(message.author, "name", "").lower():
            return

        # Check if bot is directly mentioned
        is_mentioned = bot_user in message.mentions

        # Check if message is a reply to one of the bot's messages
        is_reply_to_bot = False
        if message.reference and message.reference.message_id:
            try:
                # If resolved message exists in cache
                ref_msg = message.reference.resolved
                if ref_msg and isinstance(ref_msg, discord.Message):
                    is_reply_to_bot = ref_msg.author.id == bot_user.id
                else:
                    # Fetch referenced message if needed
                    fetched_msg = await message.channel.fetch_message(message.reference.message_id)
                    is_reply_to_bot = fetched_msg.author.id == bot_user.id
            except Exception as e:
                logger.debug("Could not resolve referenced message: %s", e)

        # 1. Team Finding Channel Handler (#find-your-team!)
        if self._is_team_finding_channel(message.channel):
            handled = await self._handle_team_finding_message(
                message=message,
                bot_user=bot_user,
                is_mentioned=is_mentioned or is_reply_to_bot,
            )
            if handled:
                return

        # 2. Author & Role validation: Only reply to 'Hacker' (participants), ignore bots & staff
        author_allowed, author_reason = self._is_author_allowed(
            author=message.author,
            bot_user=bot_user,
            is_direct_mention=is_mentioned or is_reply_to_bot,
        )
        if not author_allowed:
            logger.info("Ignoring message from %s: %s", message.author, author_reason)
            return

        # 3. Channel validation: Only reply in 'general' and 'ask-mentors' (unless directly mentioned)
        if not self._is_channel_allowed(message.channel) and not (is_mentioned or is_reply_to_bot):
            channel_name = getattr(message.channel, "name", "DM")
            logger.info(
                "Ignoring message in non-allowed channel #%s (allowed: %s)",
                channel_name,
                self.config.allowed_channel_names,
            )
            return

        cleaned_text = self._clean_content(message, bot_user)
        if not cleaned_text:
            return

        # Get recent channel context for ambiguous classifier decisions
        history_context = self.memory.get_history_context(
            channel_id=message.channel.id,
            user_id=message.author.id,
        )

        # 2. Reply Decision Layer
        should_reply, reason = await self.classifier.should_reply(
            content=cleaned_text,
            is_bot_mentioned=is_mentioned,
            is_reply_to_bot=is_reply_to_bot,
            context=history_context,
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
        if message.guild:
            core_role = discord.utils.find(
                lambda r: r.name.lower() in ["core member", "core mem"], message.guild.roles
            )
            vol_role = discord.utils.find(
                lambda r: r.name.lower() in ["volunteer", "voluntear"], message.guild.roles
            )
            if core_role:
                tags.append(core_role.mention)
            elif self.config.core_member_role_id:
                tags.append(f"<@&{self.config.core_member_role_id}>")

            if vol_role:
                tags.append(vol_role.mention)
            elif self.config.volunteer_role_id:
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
                history=history_context,
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
