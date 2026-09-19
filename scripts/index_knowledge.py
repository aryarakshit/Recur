"""CLI script to index Markdown and PDF knowledge files into FAISS."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai.embeddings import get_embedding_provider
from config import config
from database.db import Database
from rag.indexer import KnowledgeIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_knowledge")


def main() -> None:
    print("=" * 60)
    print("  HACKATHON AI AGENT - KNOWLEDGE BASE INDEXER")
    print("=" * 60)

    db = Database(config.database_path)
    embedding_provider = get_embedding_provider(config)

    indexer = KnowledgeIndexer(
        knowledge_dir=config.knowledge_dir,
        faiss_index_path=config.faiss_index_path,
        metadata_path=config.metadata_path,
        embedding_provider=embedding_provider,
        db=db,
    )

    print(f"Scanning knowledge directory: {config.knowledge_dir}")
    stats = indexer.build_index()

    print("\n--- Indexing Results ---")
    print(f"Files Indexed:      {stats.get('file_count')}")
    print(f"Chunks Created:     {stats.get('chunk_count')}")
    print(f"Vector Dimension:   {stats.get('dimension')}")
    print(f"Elapsed Time:       {stats.get('time_taken')} seconds")
    print(f"Index Saved To:     {config.faiss_index_path}")
    print(f"Metadata Saved To:  {config.metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
