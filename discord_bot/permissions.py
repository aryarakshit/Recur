"""Discord bot permissions helper for organizer commands."""

from __future__ import annotations

from typing import Union
import discord


def is_organizer(
    user_or_member: Union[discord.Member, discord.User],
    admin_role_id: int | None = None,
) -> bool:
    """Checks whether a user has organizer or administrator privileges."""
    if not isinstance(user_or_member, discord.Member):
        return False

    # Guild owner is always an organizer
    if user_or_member.guild and user_or_member.guild.owner_id == user_or_member.id:
        return True

    # Administrator permission check
    if user_or_member.guild_permissions.administrator:
        return True

    # Check configured admin_role_id
    if admin_role_id:
        role_ids = {r.id for r in user_or_member.roles}
        if admin_role_id in role_ids:
            return True

    # Check if user has a role named 'core member', 'admin', 'moderator', 'organizer', etc.
    for role in user_or_member.roles:
        if role.name.lower() in {"admin", "moderator", "core member", "volunteer", "organizer", "hackathon organizer"}:
            return True

    return False
