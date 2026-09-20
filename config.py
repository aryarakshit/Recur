"""Configuration management for Hackathon Discord AI Agent.

Loads environment variables and provides structured access to settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # Discord settings
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", "").strip())
    organizer_channel_id: int | None = field(
        default_factory=lambda: int(os.getenv("ORGANIZER_CHANNEL_ID", "0")) if os.getenv("ORGANIZER_CHANNEL_ID", "").strip().isdigit() else None
    )
    organizer_channel_name: str = field(
        default_factory=lambda: os.getenv("ORGANIZER_CHANNEL_NAME", "#help").strip()
    )
    admin_role_id: int | None = field(
        default_factory=lambda: int(os.getenv("ADMIN_ROLE_ID", "0")) if os.getenv("ADMIN_ROLE_ID", "").strip().isdigit() else None
    )
    core_member_role_id: int | None = field(
        default_factory=lambda: int(os.getenv("CORE_MEMBER_ROLE_ID", "0")) if os.getenv("CORE_MEMBER_ROLE_ID", "").strip().isdigit() else None
    )
    volunteer_role_id: int | None = field(
        default_factory=lambda: int(os.getenv("VOLUNTEER_ROLE_ID", "0")) if os.getenv("VOLUNTEER_ROLE_ID", "").strip().isdigit() else None
    )
    maintainer_role_id: int | None = field(
        default_factory=lambda: int(os.getenv("MAINTAINER_ROLE_ID", "0")) if os.getenv("MAINTAINER_ROLE_ID", "").strip().isdigit() else None
    )
    maintainer_mention: str = field(
        default_factory=lambda: os.getenv("MAINTAINER_MENTION", "@Core Member or @Volunteer").strip()
    )

    # Channel & Role Restrictions
    allowed_channel_names: list[str] = field(
        default_factory=lambda: [
            c.strip().lower().lstrip("#")
            for c in os.getenv(
                "ALLOWED_CHANNELS",
                "general,ask-mentors,ask-mentor,chat,general-chat,discussion,lounge,welcome,introductions",
            ).split(",")
            if c.strip()
        ]
    )
    allowed_role_names: list[str] = field(
        default_factory=lambda: [
            r.strip().lower()
            for r in os.getenv(
                "ALLOWED_ROLES",
                "hacker,hackers,participant,participants,attendee,attendees,student,students",
            ).split(",")
            if r.strip()
        ]
    )
    excluded_role_names: list[str] = field(
        default_factory=lambda: [
            r.strip().lower()
            for r in os.getenv(
                "EXCLUDED_ROLES",
                "admin,administrator,moderator,mod,core member,core mem,volunteer,voluntear,judge,judges,bot,bots,dyno",
            ).split(",")
            if r.strip()
        ]
    )
    moderator_mode: bool = field(
        default_factory=lambda: os.getenv("MODERATOR_MODE", "true").strip().lower() in ("true", "1", "yes")
    )
    team_finding_channel_names: list[str] = field(
        default_factory=lambda: [
            c.strip().lower().lstrip("#")
            for c in os.getenv(
                "TEAM_FINDING_CHANNELS",
                "find-your-team,find-your-team!,find-team,find-teams,find-a-team,team-finder,find-your-teammate",
            ).split(",")
            if c.strip()
        ]
    )
    team_finding_channel_id: int | None = field(
        default_factory=lambda: int(os.getenv("TEAM_FINDING_CHANNEL_ID", "0"))
        if os.getenv("TEAM_FINDING_CHANNEL_ID", "").strip().isdigit()
        else None
    )


    # LLM settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "").strip()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-004").strip()
    )

    # Paths
    base_dir: Path = BASE_DIR
    knowledge_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("KNOWLEDGE_DIR", "knowledge")
    )
    database_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("DATABASE_PATH", "data/hackbot.db")
    )
    faiss_index_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
    )
    metadata_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("METADATA_PATH", "data/metadata.json")
    )

    def __post_init__(self) -> None:
        # Auto-detect provider if not explicitly specified via LLM_PROVIDER
        if not os.getenv("LLM_PROVIDER"):
            if self.groq_api_key and not self.gemini_api_key:
                self.llm_provider = "groq"
            elif self.gemini_api_key and not self.groq_api_key:
                self.llm_provider = "gemini"

        # Default models based on provider if not specified
        if not self.llm_model:
            if self.llm_provider == "groq":
                self.llm_model = "qwen/qwen3.8-27b"
            else:
                self.llm_model = "gemini-2.5-flash"

        # Ensure directories exist
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_discord_token(self) -> bool:
        return bool(self.discord_token and self.discord_token != "replace_me")

    @property
    def active_api_key(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.gemini_api_key

    @property
    def has_active_llm_key(self) -> bool:
        key = self.active_api_key
        return bool(key and key != "replace_me")


# Global singleton instance
config = Config()
