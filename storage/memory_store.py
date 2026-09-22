"""Organizer memory saved from #recur-mem-update.

Each memory is one ``## Memory Update #N`` section in ``knowledge/memory_updates.md``.
Memories are injected into every answer prompt (not retrieved by similarity), so a
saved note keeps applying until an organizer deletes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import threading

logger = logging.getLogger(__name__)

MEMORY_FILE_NAME = "memory_updates.md"

_FILE_HEADER = (
    "# Recur Dynamic Memory & Live Organizer Updates\n"
    "*Last Updated: {timestamp}*\n\n"
    "> Official dynamic updates, announcements, and memory additions provided by organizers via #recur-mem-update.\n"
)
_SECTION_SPLIT = re.compile(r"^(?=##\s+Memory\s+Update\b)", re.MULTILINE)
_HEADER_RE = re.compile(
    r"^##\s+Memory\s+Update(?:\s+#(?P<id>\d+))?(?:\s+by\s+(?P<author>.+?))?(?:\s+\((?P<ts>[^()]*)\))?\s*$"
)
_AUTHOR_LINE_RE = re.compile(r"^-\s*\**Author\**\s*:\s*.*?\((?P<author_id>\d+)\)\s*$", re.MULTILINE)
_INFO_RE = re.compile(r"^-\s*\**Information\**\s*:[ \t]*(?P<first>.*)$", re.MULTILINE)
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "are", "was", "were", "you",
    "your", "they", "them", "their", "what", "when", "where", "which", "who", "how", "about", "into",
    "anyone", "someone", "asks", "asked", "ask", "say", "tell", "memory", "mem", "recur", "please",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPWORDS}


@dataclass
class MemoryEntry:
    id: int
    text: str
    author: str = "Organizer"
    author_id: str = ""
    timestamp: str = ""

    def preview(self, limit: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


class MemoryStore:
    def __init__(self, knowledge_dir: Path | str) -> None:
        self.path = Path(knowledge_dir) / MEMORY_FILE_NAME
        self._lock = threading.Lock()

    def list(self) -> list[MemoryEntry]:
        """Returns all memories in the order they were saved (oldest first)."""
        if not self.path.exists():
            return []
        try:
            content = self.path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Could not read %s: %s", self.path, e)
            return []

        entries: list[MemoryEntry] = []
        unnumbered: list[MemoryEntry] = []
        for section in _SECTION_SPLIT.split(content):
            lines = section.strip().splitlines()
            if not lines:
                continue
            header = _HEADER_RE.match(lines[0].strip())
            if not header:
                continue
            body = "\n".join(lines[1:])
            info = _INFO_RE.search(body)
            if info:
                rest = body[info.end():].splitlines()
                text_lines = [info.group("first")] + [ln[2:] if ln.startswith("  ") else ln for ln in rest]
                text = "\n".join(text_lines).strip()
            else:
                text = body.strip()
            if not text:
                continue
            author_line = _AUTHOR_LINE_RE.search(body)
            entry = MemoryEntry(
                id=int(header.group("id")) if header.group("id") else 0,
                text=text,
                author=(header.group("author") or "Organizer").strip(),
                author_id=author_line.group("author_id") if author_line else "",
                timestamp=(header.group("ts") or "").strip(),
            )
            entries.append(entry)
            if not entry.id:
                unnumbered.append(entry)

        # Entries written before IDs existed get the next free numbers; the next
        # write persists them, so they stay stable from then on.
        next_id = max((e.id for e in entries), default=0) + 1
        for entry in unnumbered:
            entry.id = next_id
            next_id += 1
        return entries

    def get(self, memory_id: int) -> MemoryEntry | None:
        return next((e for e in self.list() if e.id == memory_id), None)

    def add(self, text: str, author: str = "Organizer", author_id: str | int = "") -> MemoryEntry:
        with self._lock:
            entries = self.list()
            entry = MemoryEntry(
                id=max((e.id for e in entries), default=0) + 1,
                text=text.strip(),
                author=(author or "Organizer").strip(),
                author_id=str(author_id or ""),
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
            self._write(entries + [entry])
        logger.info("Saved memory #%d from %s: '%s'", entry.id, entry.author, entry.preview(60))
        return entry

    def delete(self, ids: list[int] | set[int]) -> list[MemoryEntry]:
        """Deletes memories by ID and returns the deleted entries."""
        wanted = set(ids)
        with self._lock:
            entries = self.list()
            removed = [e for e in entries if e.id in wanted]
            if removed:
                self._write([e for e in entries if e.id not in wanted])
        for e in removed:
            logger.info("Deleted memory #%d: '%s'", e.id, e.preview(60))
        return removed

    def find(self, target: str) -> list[MemoryEntry]:
        """Finds memories containing the phrase, or all of its meaningful keywords.

        Targets made only of filler words ("the", "this") match nothing, so a vague
        delete never wipes unrelated memories.
        """
        phrase = " ".join(target.lower().split())
        target_kws = _keywords(phrase)
        if not target_kws:
            return []
        phrase_re = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
        matches = []
        for e in self.list():
            text = " ".join(e.text.lower().split())
            if phrase_re.search(text) or target_kws <= _keywords(text):
                matches.append(e)
        return matches

    def is_relevant_to(self, question: str) -> bool:
        """True when any memory shares a meaningful keyword with the question."""
        q_kws = _keywords(question)
        return bool(q_kws) and any(q_kws & _keywords(e.text) for e in self.list())

    def prompt_block(self, max_chars: int = 6000) -> str:
        """Formats memories for the answer prompt, keeping the newest if over budget."""
        lines: list[str] = []
        used = 0
        for e in reversed(self.list()):
            line = f"- [#{e.id}] {' '.join(e.text.split())}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(reversed(lines))

    def _write(self, entries: list[MemoryEntry]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        parts = [_FILE_HEADER.format(timestamp=timestamp)]
        for e in entries:
            author_line = f"- **Author**: {e.author} ({e.author_id})\n" if e.author_id else f"- **Author**: {e.author}\n"
            info = "\n".join(f"  {ln}" if ln.strip() else "" for ln in e.text.splitlines())
            parts.append(
                f"\n## Memory Update #{e.id} by {e.author} ({e.timestamp})\n"
                f"{author_line}"
                f"- **Information**:\n"
                f"{info}\n"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(parts), encoding="utf-8")
