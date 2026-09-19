"""Discord message handler implementing the two-stage decision pipeline."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
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

    def _is_author_allowed(
        self,
        author: discord.User | discord.Member,
        bot_user: discord.ClientUser,
        is_direct_mention: bool = False,
    ) -> tuple[bool, str]:
        """Checks if the message author is permitted to receive AI answers.

        Rules:
        1. Always ignore bots (author.bot is True, bot_user, or name matching 'dyno').
        2. If the user DIRECTLY mentioned the bot (@Recur), always allow it (allows organizers to test).
        3. For ambient channel messages: ignore staff ('admin', 'core member', 'volunteer', 'judge', 'bot', 'dyno').
        4. Only reply to participants with the 'Hacker' / 'Participant' role.
        """
        if getattr(author, "bot", False) or author.id == bot_user.id:
            return False, "Author is a bot"

        author_name = getattr(author, "name", "").lower()
        if "dyno" in author_name:
            return False, "Author is Dyno"

        # Explicit @Recur mentions are always allowed for humans (allows admins/staff to test)
        if is_direct_mention:
            return True, "Direct mention to bot"

        # If in a guild (discord.Member), inspect roles and administrator permissions
        if hasattr(author, "roles"):
            role_names = [r.name.lower() for r in author.roles]
            perms = getattr(author, "guild_permissions", None)
            is_admin = bool(perms and getattr(perms, "administrator", False)) or any("admin" in r for r in role_names)

            # Admins are explicitly allowed so organizers can test and receive answers directly
            if is_admin:
                return True, "Author is an Admin"

            # Check for excluded staff roles (volunteer, judge, bot, dyno)
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

        return True, "Author is a participant (Hacker)"

    async def handle_message(self, message: discord.Message, bot_user: discord.ClientUser) -> None:
        """Processes an incoming message and executes the response pipeline."""
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

        # 1. Author & Role validation: Only reply to 'Hacker' (participants), ignore bots & staff
        author_allowed, author_reason = self._is_author_allowed(
            author=message.author,
            bot_user=bot_user,
            is_direct_mention=is_mentioned or is_reply_to_bot,
        )
        if not author_allowed:
            logger.info("Ignoring message from %s: %s", message.author, author_reason)
            return

        # 2. Channel validation: Only reply in 'general' and 'ask-mentors' (unless directly mentioned)
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
