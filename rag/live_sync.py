"""Live Web Sync module for Devfolio and recursiveacm.in.

Fetches the latest official hackathon updates, deadlines, registration status,
and announcements from https://recursiveacm.devfolio.co and https://recursiveacm.in,
updates the knowledge base (knowledge/live_updates.md), and triggers re-indexing.
"""

from __future__ import annotations

import datetime
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any
import urllib.request

if TYPE_CHECKING:
    from rag.indexer import KnowledgeIndexer
    from rag.retriever import KnowledgeRetriever
    from storage.database import Database

logger = logging.getLogger(__name__)


class CleanHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.hide = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ["script", "style", "head", "noscript", "svg"]:
            self.hide = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ["script", "style", "head", "noscript", "svg"]:
            self.hide = False

    def handle_data(self, data: str) -> None:
        if not self.hide:
            d = data.strip()
            if d and len(d) > 1:
                self.text.append(d)


class LiveWebSync:
    def __init__(
        self,
        knowledge_dir: str | Path,
        indexer: KnowledgeIndexer | None = None,
        retriever: KnowledgeRetriever | None = None,
        db: Database | None = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.indexer = indexer
        self.retriever = retriever
        self.db = db
        self.output_file = self.knowledge_dir / "live_updates.md"
        self.last_sync_time: float = 0.0
        self.cached_hash: str = ""
        self.cached_live_text: str = ""
        self._sync_lock = threading.Lock()

        # Load existing cached live updates if file exists
        if self.output_file.exists():
            try:
                content = self.output_file.read_text(encoding="utf-8")
                self.cached_live_text = content
                clean_initial = re.sub(r"\*Last Live Synced:[^\n]*\*", "", content).strip()
                self.cached_hash = hashlib.md5(clean_initial.encode("utf-8")).hexdigest()
            except Exception as e:
                logger.debug("Could not read initial live_updates.md: %s", e)

    def fetch_devfolio(self) -> dict[str, Any]:
        """Fetches hackathon details and schedule from Devfolio API and webpage."""
        info: dict[str, Any] = {
            "name": "RECURSIVE 2026 — Shift-8 Hackathon",
            "url": "https://recursiveacm.devfolio.co",
            "schedule_url": "https://recursiveacm.devfolio.co/schedule",
            "starts_at": "October 8, 2026",
            "ends_at": "October 8, 2026",
            "timeline": [],
            "status": "Active",
            "team_size": "2–4 members",
            "announcements": [],
        }

        # 1. Devfolio API
        try:
            req = urllib.request.Request(
                "https://api.devfolio.co/api/hackathons/recursiveacm",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RecurBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("name"):
                    info["name"] = data.get("name")
                if data.get("starts_at"):
                    info["starts_at_raw"] = data.get("starts_at")
                if data.get("ends_at"):
                    info["ends_at_raw"] = data.get("ends_at")
                if data.get("status"):
                    info["status"] = data.get("status")
                if data.get("team_min") and data.get("team_size"):
                    info["team_size"] = f"{data.get('team_min')}–{data.get('team_size')} members"
        except Exception as e:
            logger.debug("Devfolio API fetch error: %s", e)

        # 2. Devfolio Schedule page (has exact dates and milestones)
        try:
            req = urllib.request.Request(
                "https://recursiveacm.devfolio.co/schedule",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RecurBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                parser = CleanHTMLParser()
                parser.feed(html)
                text = " ".join(parser.text)
                
                # Extract date lines like '06 Sep 2026 (Sun) Registrations begin'
                matches = re.findall(
                    r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4}(?:\s+\([A-Za-z]{3}\))?\s*[^0-9\n]{0,40})",
                    text,
                )
                seen = set()
                timeline = []
                for m in matches:
                    clean_m = re.sub(r"\s+", " ", m).strip()
                    if clean_m and clean_m not in seen and len(clean_m) > 10:
                        seen.add(clean_m)
                        timeline.append(clean_m)
                if timeline:
                    info["timeline"] = timeline
        except Exception as e:
            logger.debug("Devfolio schedule fetch error: %s", e)

        return info

    def fetch_website(self) -> dict[str, Any]:
        """Fetches the latest details, announcements, and FAQs from recursiveacm.in."""
        info: dict[str, Any] = {
            "url": "https://recursiveacm.in",
            "venue": "Guru Nanak Institute of Technology, 157/F Nilgunj Road, Panihati, Sodepur, Kolkata 700114",
            "tracks": [
                "AI & Intelligent Systems",
                "FinTech & Digital Innovation",
                "HealthTech & Wellness",
                "Cybersecurity & Digital Trust",
                "Web3 & Blockchain",
                "Open Innovation",
            ],
            "announcements": [],
        }

        try:
            req = urllib.request.Request(
                "https://recursiveacm.in",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RecurBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                parser = CleanHTMLParser()
                parser.feed(html)
                lines = parser.text

                # Check for announcement or extension words
                for line in lines:
                    low = line.lower()
                    if any(term in low for term in ["extended", "extension", "deadline", "important notice", "announcement", "last date"]):
                        if len(line) > 15 and line not in info["announcements"]:
                            info["announcements"].append(line)
        except Exception as e:
            logger.debug("Website fetch error: %s", e)

        return info

    def compile_markdown(self, devfolio: dict[str, Any], website: dict[str, Any]) -> str:
        """Formats the scraped live details into clean, structured Markdown."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        timeline_lines = "\n".join(f"- {item}" for item in devfolio.get("timeline", []))
        if not timeline_lines:
            timeline_lines = (
                "- 06 Sep 2026: Registrations begin on Devfolio\n"
                "- 20 Sep 2026: Registration deadline (closes: 24 Sep 2026)\n"
                "- 08 Oct 2026: In-Person Hackathon at GNIT Kolkata (10:00 AM – 6:00 PM IST; Check-in 08:00 AM; Code Freeze 05:00 PM)"
            )

        announcements = devfolio.get("announcements", []) + website.get("announcements", [])
        announcements_block = ""
        if announcements:
            announcements_block = (
                "## Active Notices & Extension Announcements\n"
                + "\n".join(f"- 📢 **{a}**" for a in announcements)
                + "\n\n"
            )

        content = f"""# Live Web Updates from Devfolio & Official Website
*Source: {devfolio['url']} and {website['url']}*
*Last Live Synced: {now_utc}*

## Devfolio Registration, Deadlines & Timeline
- **Event**: {devfolio.get('name', 'RECURSIVE — Shift-8 Hackathon 2026')}
- **Devfolio URL**: {devfolio['url']}
- **Schedule URL**: {devfolio['schedule_url']}
- **Team Size**: {devfolio.get('team_size', '2–4 members')}
- **Devfolio Status**: {devfolio.get('status', 'Active')}

### Key Dates & Milestones (from Devfolio Schedule)
{timeline_lines}

{announcements_block}## Official Parent Website Information (https://recursiveacm.in)
- **Parent Site**: {website['url']}
- **Venue**: {website['venue']}
- **Official Tracks**: {', '.join(website.get('tracks', []))}
- **Deadlines Policy**: Registrations and project submissions are submitted on Devfolio. If an extension is granted by organizers, it will be posted on Devfolio at <https://recursiveacm.devfolio.co> and in the official Discord announcements.
"""
        return content.strip()

    def sync(self, force: bool = False) -> tuple[bool, str]:
        """Performs a live fetch and updates knowledge/live_updates.md if changed.

        Returns:
            tuple of (updated_bool, content_string)
        """
        now = time.time()
        # Enforce rate-limiting: max 1 fetch per 60 seconds unless forced
        if not force and (now - self.last_sync_time < 60.0) and self.cached_live_text:
            return False, self.cached_live_text

        with self._sync_lock:
            if not force and (now - self.last_sync_time < 60.0) and self.cached_live_text:
                return False, self.cached_live_text

            self.last_sync_time = now
            logger.info("Syncing live updates from Devfolio and recursiveacm.in...")

            devfolio_data = self.fetch_devfolio()
            website_data = self.fetch_website()
            markdown_content = self.compile_markdown(devfolio_data, website_data)

            new_hash = hashlib.md5(markdown_content.encode("utf-8")).hexdigest()
            is_new = new_hash != self.cached_hash or not self.output_file.exists()

            self.cached_live_text = markdown_content

            if is_new:
                self.cached_hash = new_hash
                try:
                    self.output_file.write_text(markdown_content, encoding="utf-8")
                    logger.info("Wrote updated live information to %s", self.output_file)

                    if self.indexer:
                        logger.info("Re-indexing knowledge base with updated live web updates...")
                        self.indexer.build_index()
                    if self.retriever:
                        self.retriever.load()
                except Exception as e:
                    logger.error("Failed to write live_updates.md or re-index: %s", e)

                return True, markdown_content

            return False, markdown_content

    def get_live_context(self, force: bool = False) -> str:
        """Retrieves the live context, refreshing if needed."""
        _, content = self.sync(force=force)
        return content

    def start_periodic_sync(self, interval_seconds: int = 900) -> None:
        """Starts a periodic background daemon thread to fetch live updates every N seconds (default: 15 mins)."""
        def _loop() -> None:
            # Wait 30 seconds after startup before initial sync
            time.sleep(30)
            while True:
                try:
                    self.sync()
                except Exception as e:
                    logger.debug("Periodic live sync error: %s", e)
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="LiveWebSyncDaemon")
        thread.start()
        logger.info("LiveWebSync daemon started (Interval: %ds).", interval_seconds)
