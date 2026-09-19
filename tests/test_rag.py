"""Unit tests for the RAG indexing and retrieval system."""

import pytest
from ai.embeddings import LocalEmbeddingProvider
from rag.indexer import KnowledgeIndexer
from rag.retriever import KnowledgeRetriever


@pytest.fixture
def temp_rag_setup(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    # Create mock markdown files
    (knowledge_dir / "rules.md").write_text(
        "# Hackathon Rules\n\n"
        "## Team Size Limits\n"
        "The maximum team size is strictly 4 members. Teams with 5 or more members are prohibited.\n\n"
        "## Original Code\n"
        "All code must be written during the event.\n",
        encoding="utf-8",
    )
    (knowledge_dir / "prizes.md").write_text(
        "# Hackathon Prizes\n\n"
        "## Cash Prizes\n"
        "1st place wins $5,000 USD. 2nd place wins $3,000 USD.\n",
        encoding="utf-8",
    )

    faiss_path = tmp_path / "test_faiss.index"
    metadata_path = tmp_path / "test_metadata.json"
    embedding_provider = LocalEmbeddingProvider(dimension=128)

    indexer = KnowledgeIndexer(
        knowledge_dir=knowledge_dir,
        faiss_index_path=faiss_path,
        metadata_path=metadata_path,
        embedding_provider=embedding_provider,
    )
    stats = indexer.build_index()

    retriever = KnowledgeRetriever(
        faiss_index_path=faiss_path,
        metadata_path=metadata_path,
        embedding_provider=embedding_provider,
    )

    return {
        "indexer": indexer,
        "retriever": retriever,
        "stats": stats,
    }


def test_indexer_builds_and_saves_index(temp_rag_setup):
    stats = temp_rag_setup["stats"]
    assert stats["file_count"] == 2
    assert stats["chunk_count"] == 3
    assert temp_rag_setup["retriever"].is_ready() is True


def test_retriever_finds_team_size_rule(temp_rag_setup):
    retriever = temp_rag_setup["retriever"]
    results = retriever.retrieve("What is the maximum team size?", top_k=2)

    assert len(results) > 0
    top_chunk = results[0].chunk
    assert top_chunk.source == "rules.md"
    assert "maximum team size is strictly 4" in top_chunk.text
    assert results[0].is_relevant is True


def test_retriever_finds_prizes(temp_rag_setup):
    retriever = temp_rag_setup["retriever"]
    results = retriever.retrieve("How much cash does 1st place win?", top_k=2)

    assert len(results) > 0
    top_chunk = results[0].chunk
    assert top_chunk.source == "prizes.md"
    assert "$5,000" in top_chunk.text


def test_retriever_unrelated_query_returns_low_or_empty(temp_rag_setup):
    retriever = temp_rag_setup["retriever"]
    results = retriever.retrieve("Quantum mechanics black holes recipe", min_score=0.35)
    assert len(results) == 0
