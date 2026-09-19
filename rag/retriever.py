"""Retriever for querying the FAISS vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from rag.models import DocumentChunk, RetrievalResult

if TYPE_CHECKING:
    from ai.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(
        self,
        faiss_index_path: Path | str,
        metadata_path: Path | str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_provider = embedding_provider
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []
        self.dimension: int = 0
        self.load()

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def load(self) -> bool:
        """Loads FAISS index and metadata from disk."""
        if not self.faiss_index_path.exists() or not self.metadata_path.exists():
            logger.warning("FAISS index or metadata file does not exist yet.")
            self.index = None
            self.chunks = []
            return False

        try:
            self.index = faiss.read_index(str(self.faiss_index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.dimension = data.get("dimension", 0)
            self.chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
            logger.info("Loaded FAISS index with %d chunks (dim=%d)", len(self.chunks), self.dimension)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            self.index = None
            self.chunks = []
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.18,
    ) -> list[RetrievalResult]:
        """Queries the FAISS index and returns top relevant chunks."""
        if not self.is_ready():
            logger.warning("Retriever requested query but index is not loaded.")
            return []

        # Generate query vector
        query_vec = self.embedding_provider.embed_query(query)
        q_matrix = np.array([query_vec], dtype=np.float32)

        # Normalize query vector
        norm = np.linalg.norm(q_matrix)
        if norm > 1e-6:
            q_matrix = q_matrix / norm

        # Search top_k nearest neighbors
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_matrix, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            # Filter low similarity results
            if float(score) >= min_score:
                results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return results

    @staticmethod
    def format_context(results: list[RetrievalResult]) -> str:
        """Formats retrieved chunks into context text for the LLM prompt."""
        if not results:
            return "No matching official hackathon context found."

        context_parts = []
        for i, res in enumerate(results, 1):
            chunk = res.chunk
            header = f"[Source {i}: {chunk.source} | Section: {chunk.section}]"
            context_parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(context_parts)
