"""Discord slash commands for organizers and administrators."""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, TYPE_CHECKING
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
    live_sync: Any | None = None,
) -> None:
    """Registers slash commands on the Discord command tree."""

    @tree.command(name="rules", description="View official hackathon rules, team limits, and judging criteria")
    async def rules_command(interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="📜 RECURSIVE 2026 — Official Rules & Guidelines",
            description="Official rules for the RECURSIVE Shift-8 Hackathon hosted by the GNIT ACM Student Chapter.",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="👥 Team Formation",
            value="• Teams must consist of **2 to 4 members**.\n• Solo participation is strictly prohibited.\n• Cross-college, cross-department, and cross-year teams are permitted.",
            inline=False,
        )
        embed.add_field(
            name="📊 Round 1 Submission (Online Idea Phase)",
            value="• Submit an **8-Slide Presentation Deck** in PDF format via Devfolio (<https://recursiveacm.devfolio.co/>).\n• Slide 9 contains template instructions and **must be deleted** prior to export.\n• Working coded prototype is NOT mandatory for Round 1, but architecture diagrams, Figma designs, or early GitHub links are highly encouraged.",
            inline=False,
        )
        embed.add_field(
            name="⚖️ Judging Rubric (100 Points)",
            value="• **Innovation & Originality**: 25%\n• **Technical Complexity**: 25%\n• **Working Prototype**: 25%\n• **UI / UX Design**: 15%\n• **Pitch & Presentation**: 10%",
            inline=False,
        )
        embed.add_field(
            name="💻 Tech Stack & AI Policy",
            value="• Freedom to choose any programming language, framework, API, or hardware platform.\n• Open-source libraries and Generative AI assistants (ChatGPT, GitHub Copilot, Claude) are permitted as accelerators, provided code is built during the hackathon and teams can defend their architecture.",
            inline=False,
        )
        embed.add_field(
            name="🏛️ Hackathon Day Sprint",
            value="• Date: **Thursday, 8 October 2026** at Guru Nanak Institute of Technology (GNIT), Kolkata.\n• Check-in: 08:00 AM – 09:30 AM IST | 8-Hour sprint concludes with code freeze at 05:00 PM.",
            inline=False,
        )
        embed.set_footer(text="GNIT ACM Student Chapter • Official Hackathon Rules")
        await interaction.response.send_message(embed=embed)

    @tree.command(name="schedule", description="View key dates, submission cutoffs, and hackathon day timeline")
    async def schedule_command(interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="🗓️ RECURSIVE 2026 — Official Timeline & Schedule",
            description="Key dates and operational schedule synced with Devfolio (<https://recursiveacm.devfolio.co/>) and official website (<https://recursiveacm.in>).",
            color=discord.Color.green(),
        )
        embed.add_field(
            name="🚀 Phase 1: Registration & Round 1 Idea Submission",
            value="• **Portal**: Devfolio (<https://recursiveacm.devfolio.co/>)\n• **Deliverable**: 8-Slide PPT Presentation (PDF)\n• **Status**: Active / In Progress\n• *Tip*: Submit 15–30 minutes early to avoid network congestion.",
            inline=False,
        )
        embed.add_field(
            name="🏛️ Phase 2: Shift-8 In-Person Hackathon Sprint",
            value="• **Date**: Thursday, 8 October 2026\n• **Venue**: Guru Nanak Institute of Technology (GNIT), 157/F Nilgunj Road, Panihati, Sodepur, Kolkata - 700114\n• **Check-in & Verification**: 08:00 AM – 09:30 AM IST (Bring College/Student ID and laptops)",
            inline=False,
        )
        embed.add_field(
            name="⏰ Hackathon Day Run of Show (8 Oct 2026)",
            value=(
                "• `08:00 AM - 09:30 AM`: Participant Arrival, Registration, Check-in & Breakfast\n"
                "• `09:30 AM - 10:00 AM`: Opening Ceremony & Track Briefing\n"
                "• `10:00 AM`: **Hacking Commences (Shift-8 Sprint Starts)** 🏁\n"
                "• `01:00 PM - 02:00 PM`: Mentorship Round 1 & Lunch Break 🥪\n"
                "• `03:30 PM - 04:30 PM`: Mentorship Round 2 & Progress Check\n"
                "• `05:00 PM`: **Code Freeze & Final Project Deployment** 🛑\n"
                "• `05:30 PM - 07:00 PM`: Final Pitching & Judging Presentations\n"
                "• `07:30 PM`: Awards Ceremony, Cash Prize Distribution & Closing 🎉"
            ),
            inline=False,
        )
        embed.set_footer(text="Devfolio Synced • GNIT ACM Student Chapter")
        await interaction.response.send_message(embed=embed)

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
            if live_sync:
                live_sync.sync(force=True)

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

    @tree.command(name="sync", description="Force an immediate live scrape of Devfolio & website and rebuild index (Organizer only)")
    async def sync_command(interaction: discord.Interaction) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            sync_updated = False
            sync_preview = ""
            if live_sync:
                sync_updated, sync_preview = live_sync.sync(force=True)

            stats = indexer.build_index()
            retriever.load()

            file_count = stats.get("file_count", 0)
            chunk_count = stats.get("chunk_count", 0)
            time_taken = stats.get("time_taken", 0.0)

            embed = discord.Embed(
                title="⚡ Live Sync & Index Rebuild Complete",
                description="Successfully fetched live data and re-indexed the vector knowledge base.",
                color=discord.Color.green(),
            )
            embed.add_field(name="Devfolio Live Scrape", value="✅ Updated" if sync_updated else "⚪ Up-to-date", inline=True)
            embed.add_field(name="Files Indexed", value=str(file_count), inline=True)
            embed.add_field(name="Chunks Generated", value=str(chunk_count), inline=True)
            embed.add_field(name="Rebuild Duration", value=f"{time_taken}s", inline=True)
            if sync_preview:
                embed.add_field(name="Live Preview", value=sync_preview[:300] + "...", inline=False)

            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("Error during /sync: %s", e)
            await interaction.followup.send(f"❌ Failed to sync and re-index: {e}", ephemeral=True)

    @tree.command(name="syncweb", description="Fetch latest live updates from Devfolio & official website (Organizer only)")
    async def syncweb_command(interaction: discord.Interaction) -> None:
        await sync_command(interaction)

    @tree.command(name="stats", description="Display operational metrics: queries, latency, and knowledge base stats (Organizer only)")
    async def stats_command(interaction: discord.Interaction) -> None:
        if not is_organizer(interaction.user, config.admin_role_id):
            await interaction.response.send_message(
                "❌ This command is restricted to hackathon organizers.",
                ephemeral=True,
            )
            return

        q_stats = database.get_query_stats()
        last_rebuild = database.get_metric("last_rebuild_time")
        file_count = database.get_metric("kb_file_count", 0)
        chunk_count = database.get_metric("kb_chunk_count", 0)

        rebuild_str = "Never"
        if last_rebuild:
            rebuild_dt = datetime.datetime.fromtimestamp(float(last_rebuild), datetime.timezone.utc)
            rebuild_str = rebuild_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        embed = discord.Embed(
            title="📊 Recur Operational Analytics & System Metrics",
            color=discord.Color.teal(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        embed.add_field(name="Total Queries Answered", value=f"**{q_stats.get('total_queries', 0)}**", inline=True)
        embed.add_field(name="Average Latency", value=f"**{q_stats.get('avg_latency', 0.0)}s**", inline=True)
        embed.add_field(name="Unanswered Queries", value=f"**{q_stats.get('unanswered_count', 0)}**", inline=True)
        embed.add_field(name="Dynamic Memory Updates", value=f"**{q_stats.get('memory_updates_count', 0)}**", inline=True)
        embed.add_field(name="Active Conversations", value=f"**{q_stats.get('active_conversations', 0)}**", inline=True)
        embed.add_field(name="Knowledge Base", value=f"**{file_count}** files / **{chunk_count}** chunks", inline=True)
        embed.add_field(name="Active LLM Model", value=f"`{config.llm_model}` ({config.llm_provider.capitalize()})", inline=True)
        embed.add_field(name="Last Re-Index", value=rebuild_str, inline=True)
        embed.add_field(
            name="Moderator Mode",
            value="🟢 Active (Admins & Moderators excluded from ambient replies)" if config.moderator_mode else "⚪ Inactive",
            inline=False,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @tree.command(name="status", description="Check bot status, LLM configuration, and knowledge base metrics")
    async def status_command(interaction: discord.Interaction) -> None:
        await stats_command(interaction)

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
        start_time = time.time()
        results = retriever.retrieve(query=question, top_k=4)
        answer, was_fallback = await generator.generate_answer(
            question=question,
            retrieval_results=results,
            organizer_channel=config.organizer_channel_name,
            organizer_tag=maintainer_str,
        )
        latency = time.time() - start_time
        try:
            database.log_query(
                question=question,
                channel_id=interaction.channel_id or 0,
                user_id=interaction.user.id,
                latency=latency,
                was_fallback=was_fallback,
            )
        except Exception as e:
            logger.debug("Could not log query in /ask: %s", e)

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
