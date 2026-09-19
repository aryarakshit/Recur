"""Data models for RAG knowledge chunks and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    section: str
    updated_at: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "updated_at": self.updated_at,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            section=data.get("section", "General"),
            updated_at=data.get("updated_at", ""),
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float

    @property
    def is_relevant(self) -> bool:
        # Cosine similarity score threshold
        return self.score >= 0.18
