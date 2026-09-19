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

    async def handle_message(self, message: discord.Message, bot_user: discord.ClientUser) -> None:
        """Processes an incoming message and executes the response pipeline."""
        # 1. Ignore messages from bots (including self)
        if message.author.bot or message.author.id == bot_user.id:
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
                await message.reply(answer, mention_author=False)
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
