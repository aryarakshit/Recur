"""Discord slash commands for organizers and administrators."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING
import discord
from discord import app_commands

from discord_bot.permissions import is_organizer

if TYPE_CHECKING:
    from ai.generator import AnswerGenerator
    from config import Config
    from rag.indexer import KnowledgeIndexer
    from rag.retriever import KnowledgeRetriever
    from storage.database import Database

logger = logging.getLogger(__name__)


def setup_commands(
    tree: app_commands.CommandTree,
    indexer: KnowledgeIndexer,
    retriever: KnowledgeRetriever,
    generator: AnswerGenerator,
    database: Database,
    config: Config,
) -> None:
    """Registers slash commands on the Discord command tree."""

    @tree.command(name="reloadkb", description="Reload and re-index the official knowledge base (Organizer only)")
    async def reloadkb_command(interaction: discord.Interaction) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            stats = indexer.build_index()
            # Reload retriever index in memory
            retriever.load()

            file_count = stats.get("file_count", 0)
            chunk_count = stats.get("chunk_count", 0)
            time_taken = stats.get("time_taken", 0.0)

            embed = discord.Embed(
                title="✅ Knowledge Base Reloaded",
                description=f"Successfully scanned and re-indexed the knowledge base in **{time_taken}s**.",
                color=discord.Color.green(),
            )
            embed.add_field(name="Files Indexed", value=str(file_count), inline=True)
            embed.add_field(name="Chunks Generated", value=str(chunk_count), inline=True)
            embed.add_field(name="Embedding Provider", value=indexer.embedding_provider.__class__.__name__, inline=True)

            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("Error during /reloadkb: %s", e)
            await interaction.followup.send(f"❌ Failed to reload knowledge base: {e}", ephemeral=True)

    @tree.command(name="status", description="Check bot status, LLM configuration, and knowledge base metrics")
    async def status_command(interaction: discord.Interaction) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        last_rebuild = database.get_metric("last_rebuild_time")
        file_count = database.get_metric("kb_file_count", 0)
        chunk_count = database.get_metric("kb_chunk_count", 0)

        rebuild_str = "Never"
        if last_rebuild:
            rebuild_dt = datetime.datetime.fromtimestamp(float(last_rebuild), datetime.timezone.utc)
            rebuild_str = rebuild_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        embed = discord.Embed(
            title="🤖 Hackathon AI Agent Status",
            color=discord.Color.blue(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        embed.add_field(name="Bot Status", value="🟢 Online", inline=True)
        embed.add_field(name="Configured Provider", value=config.llm_provider.capitalize(), inline=True)
        embed.add_field(name="Active Model", value=config.llm_model or "Default", inline=True)
        embed.add_field(name="KB Files", value=str(file_count), inline=True)
        embed.add_field(name="KB Chunks", value=str(chunk_count), inline=True)
        embed.add_field(name="Last Rebuild", value=rebuild_str, inline=True)
        embed.add_field(
            name="Moderator Mode",
            value="🟢 Active (Admins & Moderators excluded from ambient replies)" if config.moderator_mode else "⚪ Inactive",
            inline=False,
        )
        embed.add_field(
            name="API Key Configured",
            value="✅ Yes" if config.has_active_llm_key else "⚠️ Blank (Offline Mock Mode)",
            inline=False,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @tree.command(name="asktest", description="Test a question against the knowledge base privately (Organizer only)")
    @app_commands.describe(question="The question to test")
    async def asktest_command(interaction: discord.Interaction, question: str) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        results = retriever.retrieve(query=question, top_k=4)
        answer, was_fallback = await generator.generate_answer(
            question=question,
            retrieval_results=results,
            organizer_channel=config.organizer_channel_name,
        )

        embed = discord.Embed(
            title="🧪 Knowledge Base Test Query",
            color=discord.Color.gold() if was_fallback else discord.Color.green(),
        )
        embed.add_field(name="Question", value=question, inline=False)
        embed.add_field(name="Generated Answer", value=answer, inline=False)
        embed.add_field(name="Result Type", value="⚠️ Safe Fallback" if was_fallback else "✅ Grounded Answer", inline=True)

        sources_summary = []
        for r in results:
            sources_summary.append(f"• **{r.chunk.source}** ({r.chunk.section}) - Score: `{r.score:.3f}`")

        embed.add_field(
            name="Retrieved Sources",
            value="\n".join(sources_summary) if sources_summary else "None",
            inline=False,
        )

        await interaction.followup.send(embed=embed, ephemeral=True)

    @tree.command(name="ask", description="Manually ask the bot a question from the knowledge base")
    @app_commands.describe(question="The question to ask")
    async def ask_command(interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer()

        # Tag maintainer roles if fallback occurs
        maintainer_str = config.maintainer_mention or "@Core Member or @Volunteer"
        results = retriever.retrieve(query=question, top_k=4)
        answer, was_fallback = await generator.generate_answer(
            question=question,
            retrieval_results=results,
            organizer_channel=config.organizer_channel_name,
            organizer_tag=maintainer_str,
        )

        if was_fallback:
            sources = [r.chunk.source for r in results]
            database.log_unanswered_question(
                question=question,
                channel_id=interaction.channel_id or 0,
                user_id=interaction.user.id,
                retrieved_sources=sources,
            )

        await interaction.followup.send(answer)

    @tree.command(name="clearcache", description="Clear temporary conversation history cache (Admin only)")
    async def clearcache_command(interaction: discord.Interaction) -> None:
        if not (interaction.user.guild_permissions.administrator or is_organizer(interaction.user, config.admin_role_id)):
            await interaction.response.send_message(
                "❌ This command requires Discord administrator permissions.",
                ephemeral=True,
            )
            return

        deleted = database.clear_conversation_cache()
        await interaction.response.send_message(
            f"🧹 Cleared {deleted} conversation memory records from the cache.",
            ephemeral=True,
        )

    @tree.command(name="unanswered", description="Review recently unanswered questions logged by the bot")
    @app_commands.describe(limit="Number of questions to view (default 10)")
    async def unanswered_command(interaction: discord.Interaction, limit: int = 10) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        unanswered = database.get_unanswered_questions(limit=min(limit, 25))
        if not unanswered:
            await interaction.response.send_message("🎉 No unanswered questions currently logged!", ephemeral=True)
            return

        embed = discord.Embed(
            title="📋 Unanswered Questions Log",
            description="These questions triggered the fallback and may need documentation in `knowledge/*.md`:",
            color=discord.Color.orange(),
        )

        for item in unanswered[:10]:
            dt = datetime.datetime.fromtimestamp(item["timestamp"], datetime.timezone.utc).strftime("%m/%d %H:%M")
            embed.add_field(
                name=f"[{dt}] Question #{item['id']}",
                value=f"**User**: <@{item['user_id']}>\n**Query**: {item['question']}",
                inline=False,
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)
