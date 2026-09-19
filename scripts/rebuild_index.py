"""CLI script to build or rebuild the FAISS knowledge base index."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai.embeddings import get_embedding_provider
from config import config
from rag.indexer import KnowledgeIndexer
from storage.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_index")


def main() -> None:
    print("=" * 60)
    print("  HACKATHON AI AGENT - KNOWLEDGE BASE INDEX REBUILD")
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
