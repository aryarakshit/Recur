"""Retriever for querying the FAISS vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import faiss
import numpy as np

from rag.models import DocumentChunk, RetrievalResult

if TYPE_CHECKING:
    from ai.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "s", "same", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "recur"
}


def normalize_query_text(query: str) -> str:
    """Normalizes conversational queries and common hackathon typos."""
    clean = query.strip()
    clean = re.sub(r"^(?:(?:so|hey|hi|yo|ok|okay)\s+)?(?:@?recur\s*[,:]?\s*)", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bpricepool\b", "prize pool", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bprice\s+pool\b", "prize pool", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bprices\b", "prizes", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bteamates\b", "teammates", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bteamate\b", "teammate", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsubmition\b", "submission", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bregestration\b", "registration", clean, flags=re.IGNORECASE)
    return clean.strip() or query.strip()


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
        min_score: float = 0.05,
    ) -> list[RetrievalResult]:
        """Queries the FAISS index using hybrid dense vector + lexical reranking, prioritizing live memory updates."""
        if not self.is_ready():
            logger.warning("Retriever requested query but index is not loaded.")
            return []

        norm_query = normalize_query_text(query)

        # Extract content keywords for lexical overlap
        words = re.findall(r"\b[a-z0-9_]{3,}\b", norm_query.lower())
        content_keywords = {w for w in words if w not in COMMON_STOPWORDS}

        # Generate query vector
        query_vec = self.embedding_provider.embed_query(norm_query)
        q_matrix = np.array([query_vec], dtype=np.float32)

        # Normalize query vector
        norm = np.linalg.norm(q_matrix)
        if norm > 1e-6:
            q_matrix = q_matrix / norm

        # Search larger candidate pool to prevent relevant chunks from being cut off by stop-word noise
        num_candidates = min(len(self.chunks), max(top_k * 5, 25))
        if self.index.ntotal > 0:
            scores, indices = self.index.search(q_matrix, min(num_candidates, self.index.ntotal))
            candidate_indices = set(indices[0]) if len(indices) > 0 else set()
        else:
            candidate_indices = set()

        # Always include any live memory_updates chunks in evaluation
        for idx, c in enumerate(self.chunks):
            if "memory_updates" in c.source.lower():
                candidate_indices.add(idx)

        # Compute hybrid score for all candidates
        scored_candidates: list[tuple[float, DocumentChunk]] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            # Vector similarity
            c_vec = self.embedding_provider.embed_query(chunk.text)
            c_matrix = np.array([c_vec], dtype=np.float32)
            c_norm = np.linalg.norm(c_matrix)
            v_score = float(np.dot(q_matrix[0], c_matrix[0]) / (norm * c_norm)) if (norm > 1e-6 and c_norm > 1e-6) else 0.0

            # Lexical keyword matching
            chunk_text_lower = (chunk.source + " " + chunk.section + " " + chunk.text).lower()
            keyword_matches = 0.0
            if content_keywords:
                for kw in content_keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", chunk_text_lower):
                        keyword_matches += 1.0
                    elif kw in chunk_text_lower:
                        keyword_matches += 0.5
                kw_score = keyword_matches / len(content_keywords)
            else:
                kw_score = 0.0

            # Live organizer memory boost:
            # If an organizer wrote a live update in #recur-mem-update, it takes priority
            is_mem = "memory_updates" in chunk.source.lower()
            mem_boost = 0.15 if is_mem else 0.0
            if is_mem and kw_score > 0:
                mem_boost += 0.20

            final_score = (0.5 * v_score) + (0.5 * kw_score) + mem_boost

            if final_score >= min_score or (is_mem and kw_score > 0):
                scored_candidates.append((final_score, chunk))

        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievalResult] = [
            RetrievalResult(chunk=c, score=s)
            for s, c in scored_candidates[:top_k]
        ]
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
