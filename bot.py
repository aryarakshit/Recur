"""Main entry point for Hackathon Discord AI Agent."""

from __future__ import annotations

import logging
import sys
import discord
from discord.ext import commands

from ai.classifier import MessageClassifier
from ai.embeddings import get_embedding_provider
from ai.generator import AnswerGenerator
from ai.provider import get_llm_provider
from config import config
from discord_bot.commands import setup_commands
from discord_bot.message_handler import MessageHandler
from rag.indexer import KnowledgeIndexer
from rag.retriever import KnowledgeRetriever
from database.db import Database
from storage.memory import ConversationMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Recur")


def print_missing_token_guide() -> None:
    print("\n" + "=" * 65)
    print(" [!] DISCORD_TOKEN IS BLANK IN .env")
    print("=" * 65)
    print(" The bot initialized successfully, but needs a Discord token to log in.")
    print("\n Setup Instructions:")
    print(" 1. Go to the Discord Developer Portal:")
    print("    https://discord.com/developers/applications")
    print(" 2. Select or create your Application and navigate to 'Bot'.")
    print(" 3. Under 'Privileged Gateway Intents', ENABLE:")
    print("    * Message Content Intent")
    print(" 4. Click 'Reset Token' and copy your bot token.")
    print(" 5. Open '.env' in this folder and set:")
    print("    DISCORD_TOKEN=your_token_here")
    print(" 6. Add your LLM key when ready (GEMINI_API_KEY or GROQ_API_KEY).")
    print(" 7. Run 'python bot.py' again.")
    print("=" * 65 + "\n")


def main() -> None:
    print("Initializing Hackathon Discord AI Agent...")

    # Initialize SQLite Database & Metrics
    db = Database(config.database_path)

    # Initialize Embedding Provider & RAG
    embedding_provider = get_embedding_provider(config)
    indexer = KnowledgeIndexer(
        knowledge_dir=config.knowledge_dir,
        faiss_index_path=config.faiss_index_path,
        metadata_path=config.metadata_path,
        embedding_provider=embedding_provider,
        db=db,
    )
    retriever = KnowledgeRetriever(
        faiss_index_path=config.faiss_index_path,
        metadata_path=config.metadata_path,
        embedding_provider=embedding_provider,
    )

    # If FAISS index doesn't exist yet, build it automatically
    if not retriever.is_ready():
        logger.info("Knowledge base index not found. Building initial index...")
        indexer.build_index()
        retriever.load()

    # Initialize LLM & Decision Pipeline
    llm_provider = get_llm_provider(config)
    classifier = MessageClassifier(llm_provider=llm_provider)
    generator = AnswerGenerator(
        llm_provider=llm_provider,
        default_organizer_channel=config.organizer_channel_name,
        classifier=classifier,
    )
    memory = ConversationMemory(db=db, max_history_turns=6)
    message_handler = MessageHandler(
        classifier=classifier,
        generator=generator,
        retriever=retriever,
        memory=memory,
        database=db,
        config=config,
    )

    # Check for Discord Token
    if not config.has_discord_token:
        print_missing_token_guide()
        sys.exit(0)

    # Configure Discord client with required intents
    intents = discord.Intents.default()
    intents.message_content = True  # Required to inspect message text

    bot = commands.Bot(command_prefix="!", intents=intents)

    # Setup slash commands
    setup_commands(
        tree=bot.tree,
        indexer=indexer,
        retriever=retriever,
        generator=generator,
        database=db,
        config=config,
    )

    @bot.event
    async def on_ready() -> None:
        logger.info("Logged in as %s (ID: %s)", bot.user.name, bot.user.id)
        guilds = [f"'{g.name}' (ID: {g.id}, Members: {g.member_count})" for g in bot.guilds]
        logger.info("Connected to %d guild(s): %s", len(guilds), ", ".join(guilds) if guilds else "None")
        try:
            synced = await bot.tree.sync()
            logger.info("Synced %d slash commands across Discord.", len(synced))
        except Exception as e:
            logger.error("Failed to sync slash commands: %s", e)

        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"hackathon questions | {config.organizer_channel_name}",
        )
        await bot.change_presence(activity=activity)
        logger.info("Bot is ready and listening for hackathon queries!")

        for guild in bot.guilds:
            logger.info("Examining channels in '%s':", guild.name)
            for ch in guild.text_channels:
                perms = ch.permissions_for(guild.me)
                logger.info("  #%s: send_messages=%s, view_channel=%s", ch.name, perms.send_messages, perms.view_channel)

    @bot.event
    async def on_message(message: discord.Message) -> None:
        channel_name = getattr(message.channel, "name", "DM")
        logger.info("Message received in #%s from %s: '%s'", channel_name, message.author, message.content)
        await message_handler.handle_message(message=message, bot_user=bot.user)

    try:
        bot.run(config.discord_token)
    except discord.errors.LoginFailure:
        logger.critical("Failed to log in: Invalid DISCORD_TOKEN provided in .env")
    except Exception as e:
        logger.critical("Unexpected error while running bot: %s", e)


if __name__ == "__main__":
    main()
