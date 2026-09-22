"""Main entry point for Hackathon Discord AI Agent."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import asyncio
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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

import collections
import json
import re

# In-memory log buffer for cloud telemetry and diagnostics
_log_buffer: collections.deque[str] = collections.deque(maxlen=100)
_app_state: dict[str, object] = {
    "status": "initializing",
    "start_time": time.time(),
    "bot": None,
    "last_error": None,
}


class BufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            msg = re.sub(r"MTU[a-zA-Z0-9_\-\.]+", "[MASKED_TOKEN]", msg)
            msg = re.sub(r"gsk_[a-zA-Z0-9]+", "[MASKED_GROQ]", msg)
            msg = re.sub(r"AIza[a-zA-Z0-9_\-]+", "[MASKED_GEMINI]", msg)
            _log_buffer.append(msg)
        except Exception:
            pass


buffer_handler = BufferLogHandler()
buffer_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), buffer_handler],
)
logger = logging.getLogger("Recur")


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler to satisfy external monitors & Render health checks."""

    def do_HEAD(self) -> None:
        body = b"Recur Bot is running and healthy!\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/status", "/healthz", "/debug", "/logs", "/json"):
            bot_inst = _app_state.get("bot")
            is_ready = bool(bot_inst and getattr(bot_inst, "is_ready", lambda: False)())
            bot_user = str(getattr(bot_inst, "user", None)) if bot_inst else None
            guild_list = (
                [f"{g.name} (ID: {g.id})" for g in getattr(bot_inst, "guilds", [])]
                if bot_inst and getattr(bot_inst, "guilds", None)
                else []
            )
            data = {
                "service_status": _app_state.get("status", "unknown"),
                "bot_ready": is_ready,
                "bot_user": bot_user,
                "guilds": guild_list,
                "uptime_seconds": int(time.time() - _app_state.get("start_time", time.time())),
                "has_discord_token": config.has_discord_token,
                "token_prefix": (config.discord_token[:6] + "...") if config.discord_token else None,
                "llm_provider": config.llm_provider,
                "allowed_channels": config.allowed_channel_names,
                "last_error": str(_app_state.get("last_error")) if _app_state.get("last_error") else None,
                "recent_logs": list(_log_buffer)[-50:],
            }
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return

        body = b"Recur Bot is running and healthy!\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


def start_health_server() -> None:
    port = int(os.getenv("PORT", "7860"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), HealthCheckHandler)
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
        db=db,
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
        indexer=indexer,
    )

    # Check for Discord Token
    if not config.has_discord_token:
        print_missing_token_guide()
        sys.exit(0)

    # Configure Discord client with required intents and allowed mentions
    intents = discord.Intents.default()
    intents.message_content = True  # Required to inspect message text (Privileged intent enabled in portal)
    allowed_mentions = discord.AllowedMentions(everyone=True, users=True, roles=True, replied_user=True)

    bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=allowed_mentions)
    _app_state["bot"] = bot
    startup_catchup_done = False
    commands_synced = False
    restart_delay = 5.0

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
        nonlocal commands_synced, startup_catchup_done, restart_delay
        restart_delay = 5.0
        _app_state["status"] = "online"
        logger.info("Recur is online.")
        logger.info("Logged in as %s (ID: %s)", bot.user.name, bot.user.id)
        guilds = [f"'{g.name}' (ID: {g.id}, Members: {g.member_count})" for g in bot.guilds]
        logger.info("Connected to %d guild(s): %s", len(guilds), ", ".join(guilds) if guilds else "None")
        if not commands_synced:
            try:
                synced = await bot.tree.sync()
                commands_synced = True
                logger.info("Synced %d slash commands across Discord.", len(synced))
            except Exception as e:
                logger.error("Failed to sync slash commands: %s", e)

        activity_name = os.getenv("BOT_ACTIVITY", "hackathon questions | help desk")
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=activity_name,
        )
        await bot.change_presence(activity=activity)
        logger.info("Bot is ready and listening for hackathon queries!")

        # Catch up on any unanswered messages while bot was offline / restarting
        if not startup_catchup_done:
            try:
                logger.info("Scanning for any unanswered messages in allowed channels...")
                caught_up = await message_handler.catch_up_unanswered_messages(
                    bot_user=bot.user,
                    guilds=bot.guilds,
                    limit_per_channel=25,
                )
                startup_catchup_done = True
                logger.info("Catch-up completed: %d unanswered message(s) processed.", caught_up)
            except Exception as e:
                logger.error("Error during startup catch-up: %s", e)

        # Start periodic catch-up task to prevent unanswered queries
        if not periodic_catch_up.is_running():
            periodic_catch_up.start()

        # Start scheduled daily memory self-update task
        if not daily_memory_sync.is_running():
            daily_memory_sync.start()

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

    @tasks.loop(hours=24)
    async def daily_memory_sync() -> None:
        """Daily 24-hour maintenance task to refresh Devfolio & website details and log audit metrics."""
        try:
            logger.info("Executing scheduled daily memory refresh from Devfolio & website...")
            updated, _ = live_sync.sync(force=True)
            db.set_metric("daily_sync_status", {
                "timestamp": time.time(),
                "status": "success",
                "updated": updated,
            })
            logger.info("Scheduled daily memory sync completed (Updated: %s).", updated)
        except Exception as e:
            logger.error("Error in scheduled daily memory sync: %s", e)

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

    async def run_forever() -> None:
        nonlocal restart_delay
        while True:
            try:
                await bot.start(config.discord_token, reconnect=True)
                logger.warning("Discord client stopped; retrying without replacing the process.")
            except discord.errors.LoginFailure as e:
                _app_state["last_error"] = f"LoginFailure: {e}"
                _app_state["status"] = "login_failure"
                logger.critical("Fatal: Invalid DISCORD_TOKEN provided. Please check environment variables: %s", e)
                raise
            except discord.errors.HTTPException as e:
                _app_state["last_error"] = f"HTTPException {getattr(e, 'status', 'unknown')}: {e}"
                _app_state["status"] = "rate_limited" if getattr(e, "status", None) == 429 else "http_error"
                logger.error("Discord HTTP Exception: %s", e)
                if getattr(e, "status", None) == 429:
                    retry_after = getattr(e, "retry_after", None)
                    restart_delay = max(float(retry_after or 0), 300.0, restart_delay)
                    logger.warning(
                        "Discord Cloudflare Rate Limit (HTTP 429, code 0). Backing off for %.0fs before retrying...",
                        restart_delay,
                    )
                else:
                    restart_delay = min(max(restart_delay, 15.0) * 2, 900.0)
            except Exception as e:
                _app_state["last_error"] = f"{type(e).__name__}: {e}"
                _app_state["status"] = "error"
                restart_delay = min(max(restart_delay, 15.0) * 2, 900.0)
                logger.error("Unexpected error in Discord bot runner: %s", e, exc_info=True)
            finally:
                if not bot.is_closed():
                    await bot.close()

            await asyncio.sleep(restart_delay)

    try:
        asyncio.run(run_forever())
    except discord.errors.LoginFailure:
        time.sleep(30)
        sys.exit(1)


if __name__ == "__main__":
    main()
