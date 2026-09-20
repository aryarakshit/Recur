"""Main entry point for Hackathon Discord AI Agent."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands, tasks

from ai.classifier import MessageClassifier
from ai.embeddings import get_embedding_provider
from ai.generator import AnswerGenerator
from ai.provider import get_llm_provider
from config import config
from discord_bot.commands import setup_commands
from discord_bot.message_handler import MessageHandler
from rag.indexer import KnowledgeIndexer
from rag.retriever import KnowledgeRetriever
from rag.live_sync import LiveWebSync
from storage.database import Database
from storage.memory import ConversationMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Recur")


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler to satisfy Hugging Face Spaces & Render health checks."""

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Recur Bot is running and healthy!\n")

    def log_message(self, format: str, *args: object) -> None:
        pass


def start_health_server() -> None:
    port = int(os.getenv("PORT", "7860"))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("Health check server listening on 0.0.0.0:%d", port)
    except Exception as e:
        logger.warning("Could not bind health server on port %d: %s", port, e)


def start_keep_alive() -> None:
    """Sends a periodic HTTP GET request to the public URL to prevent Render from idling/sleeping."""
    import urllib.request

    def _ping_loop() -> None:
        url = os.getenv("RENDER_EXTERNAL_URL")
        if not url:
            return
        logger.info("Render external URL detected: %s. Starting keep-alive self-ping loop.", url)
        # Wait 3 minutes after startup before first ping
        time.sleep(180)
        while True:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RecurKeepAlive/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    logger.info("Keep-alive ping to %s succeeded (HTTP %d)", url, resp.status)
            except Exception as e:
                logger.debug("Keep-alive ping to %s: %s", url, e)
            # Sleep 9 minutes (Render idle timeout is 15 minutes)
            time.sleep(540)

    thread = threading.Thread(target=_ping_loop, daemon=True)
    thread.start()


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

    # Start lightweight health-check HTTP server for Hugging Face Spaces / Render
    start_health_server()
    start_keep_alive()

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

    # Initialize Live Web Sync for Devfolio and recursiveacm.in
    live_sync = LiveWebSync(
        knowledge_dir=config.knowledge_dir,
        indexer=indexer,
        retriever=retriever,
    )
    live_sync.start_periodic_sync(interval_seconds=900)

    # Initialize LLM & Decision Pipeline
    llm_provider = get_llm_provider(config)
    classifier = MessageClassifier(llm_provider=llm_provider)
    generator = AnswerGenerator(
        llm_provider=llm_provider,
        default_organizer_channel=config.organizer_channel_name,
        classifier=classifier,
        live_sync=live_sync,
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

    # Configure Discord client with required intents and allowed mentions
    intents = discord.Intents.default()
    intents.message_content = True  # Required to inspect message text
    allowed_mentions = discord.AllowedMentions(everyone=True, users=True, roles=True, replied_user=True)

    bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=allowed_mentions)

    # Setup slash commands
    setup_commands(
        tree=bot.tree,
        indexer=indexer,
        retriever=retriever,
        generator=generator,
        database=db,
        config=config,
        live_sync=live_sync,
    )

    @bot.event
    async def on_ready() -> None:
        logger.info("Recur is online.")
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

        # Catch up on any unanswered messages while bot was offline / restarting
        try:
            logger.info("Scanning for any unanswered messages in allowed channels...")
            caught_up = await message_handler.catch_up_unanswered_messages(
                bot_user=bot.user,
                guilds=bot.guilds,
                limit_per_channel=25,
            )
            logger.info("Catch-up completed: %d unanswered message(s) processed.", caught_up)
        except Exception as e:
            logger.error("Error during startup catch-up: %s", e)

        # Start periodic catch-up task to prevent unanswered queries
        if not periodic_catch_up.is_running():
            periodic_catch_up.start()

    @tasks.loop(minutes=5)
    async def periodic_catch_up() -> None:
        """Periodic background task to catch up on any missed or unanswered messages."""
        try:
            count = await message_handler.catch_up_unanswered_messages(
                bot_user=bot.user,
                guilds=bot.guilds,
                limit_per_channel=15,
            )
            if count > 0:
                logger.info("Periodic catch-up: processed %d unanswered message(s).", count)
        except Exception as e:
            logger.debug("Error in periodic catch-up task: %s", e)

    @bot.event
    async def on_resumed() -> None:
        logger.info("Discord session resumed successfully. Recur is online.")

    @bot.event
    async def on_disconnect() -> None:
        logger.warning("Discord gateway disconnected. Reconnecting automatically...")

    @bot.event
    async def on_message(message: discord.Message) -> None:
        try:
            channel_name = getattr(message.channel, "name", "DM")
            logger.info("Message received in #%s from %s: '%s'", channel_name, message.author, message.content)
            await message_handler.handle_message(message=message, bot_user=bot.user)
        except Exception as e:
            logger.error("Error processing message '%s': %s", getattr(message, "content", ""), e, exc_info=True)

    # Automatic reconnection loop
    retry_delay = 5
    while True:
        try:
            bot.run(config.discord_token, reconnect=True)
            break
        except discord.errors.LoginFailure:
            logger.critical("Fatal: Invalid DISCORD_TOKEN provided. Please check environment variables.")
            sys.exit(1)
        except (discord.errors.GatewayNotFound, discord.errors.ConnectionClosed) as e:
            logger.warning("Discord connection error: %s. Reconnecting in %ds...", e, retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
        except Exception as e:
            logger.error("Unexpected error in Discord bot runner: %s. Retrying in %ds...", e, retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


if __name__ == "__main__":
    main()
