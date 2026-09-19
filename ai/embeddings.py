"""Embedding generation for Hackathon Discord AI Agent.

Supports:
1. Gemini Embeddings via google-genai SDK (`text-embedding-004`)
2. Local deterministic subword/n-gram hashing vectorizer for zero-dependency offline/blank-key operation
"""

from __future__ import annotations

import abc
import hashlib
import logging
import re
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class EmbeddingProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        pass

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronously compute embeddings for a batch of texts."""
        pass

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously compute embeddings."""
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Compute embedding for a single query."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self.dimension


class LocalEmbeddingProvider(EmbeddingProvider):
    """Deterministic, zero-dependency sub-word n-gram vectorizer.

    Generates L2-normalized float32 vectors suitable for FAISS cosine similarity.
    Works offline, instantly, and requires no API keys or downloaded models.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _extract_features(self, text: str) -> list[str]:
        # Lowercase and clean
        text = text.lower()
        words = re.findall(r"\b[a-z0-9_]{2,}\b", text)
        features = list(words)

        # Word bi-grams
        for i in range(len(words) - 1):
            features.append(f"{words[i]}_{words[i+1]}")

        # Char 3-grams for words to capture stem/typo variations
        for w in words:
            if len(w) >= 3:
                for j in range(len(w) - 2):
                    features.append(f"#{w[j:j+3]}")

        return features

    def _vectorize(self, text: str) -> list[float]:
        features = self._extract_features(text)
        vec = np.zeros(self._dim, dtype=np.float32)

        if not features:
            return vec.tolist()

        for feat in features:
            # Hash feature to index and sign
            h = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return vec.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini API embedding generator with local fallback."""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        self.api_key = api_key
        self.model = model or "text-embedding-004"
        self._fallback = LocalEmbeddingProvider(dimension=768)
        self._client = None
        if api_key and api_key != "replace_me":
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.warning("Failed to initialize google-genai client: %s. Using fallback.", e)

    @property
    def dimension(self) -> int:
        return 768

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            return self._fallback.embed_texts(texts)

        try:
            from google.genai import types
            embeddings = []
            # Batch call to Gemini embedding
            for chunk_batch in [texts[i:i + 10] for i in range(0, len(texts), 10)]:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=chunk_batch,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                for emb in response.embeddings:
                    vec = np.array(emb.values, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    embeddings.append(vec.tolist())
            return embeddings
        except Exception as e:
            logger.warning("Gemini embedding API call failed: %s. Falling back to local vectorizer.", e)
            return self._fallback.embed_texts(texts)


def get_embedding_provider(config: Config) -> EmbeddingProvider:
    """Factory to retrieve the appropriate embedding provider."""
    if config.gemini_api_key and config.gemini_api_key != "replace_me":
        logger.info("Using Gemini Embedding Provider (%s)", config.embedding_model)
        return GeminiEmbeddingProvider(
            api_key=config.gemini_api_key,
            model=config.embedding_model,
        )
    logger.info("API key not configured for embeddings. Using LocalEmbeddingProvider.")
    return LocalEmbeddingProvider(dimension=384)
