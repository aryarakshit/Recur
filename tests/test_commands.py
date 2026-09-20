"""Unit tests for Discord slash commands."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import discord
from discord import app_commands

from config import Config
from discord_bot.commands import setup_commands
from storage.database import Database


@pytest.fixture
def mock_tree():
    tree = MagicMock(spec=app_commands.CommandTree)
    registered_commands = {}

    def command(name=None, description=None, **kwargs):
        def decorator(func):
            cmd_name = name or func.__name__
            registered_commands[cmd_name] = func
            return func
        return decorator

    tree.command = command
    tree._registered = registered_commands
    return tree


@pytest.fixture
def test_setup(tmp_path, mock_tree):
    db = Database(tmp_path / "test.db")
    config = Config()
    indexer = MagicMock()
    retriever = MagicMock()
    generator = MagicMock()
    live_sync = MagicMock()

    setup_commands(
        tree=mock_tree,
        indexer=indexer,
        retriever=retriever,
        generator=generator,
        database=db,
        config=config,
        live_sync=live_sync,
    )
    return {
        "commands": mock_tree._registered,
        "database": db,
        "config": config,
        "indexer": indexer,
        "retriever": retriever,
        "generator": generator,
        "live_sync": live_sync,
    }


def test_registered_command_names(test_setup):
    cmds = test_setup["commands"]
    assert "rules" in cmds
    assert "schedule" in cmds
    assert "sync" in cmds
    assert "stats" in cmds
    assert "ask" in cmds
    assert "reloadkb" in cmds
    assert "status" in cmds


@pytest.mark.asyncio
async def test_rules_command(test_setup):
    rules_cmd = test_setup["commands"]["rules"]
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.response = AsyncMock()

    await rules_cmd(interaction)
    interaction.response.send_message.assert_called_once()
    embed = interaction.response.send_message.call_args.kwargs.get("embed")
    assert embed is not None
    assert "Rules & Guidelines" in embed.title
    field_names = [f.name for f in embed.fields]
    assert any("Team Formation" in n for n in field_names)
    assert any("Judging Rubric" in n for n in field_names)


@pytest.mark.asyncio
async def test_schedule_command(test_setup):
    schedule_cmd = test_setup["commands"]["schedule"]
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.response = AsyncMock()

    await schedule_cmd(interaction)
    interaction.response.send_message.assert_called_once()
    embed = interaction.response.send_message.call_args.kwargs.get("embed")
    assert embed is not None
    assert "Timeline & Schedule" in embed.title
    field_names = [f.name for f in embed.fields]
    assert any("Phase 1" in n for n in field_names)
    assert any("Phase 2" in n for n in field_names)
