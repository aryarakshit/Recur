"""Unit tests for LiveWebSync and deadline extension handling."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ai.generator import AnswerGenerator
from ai.provider import MockProvider
from rag.live_sync import LiveWebSync
from rag.models import DocumentChunk, RetrievalResult


def test_compile_markdown_formats_dates_and_announcements(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    syncer = LiveWebSync(knowledge_dir=knowledge_dir)

    devfolio_data = {
        "name": "RECURSIVE 2026",
        "url": "https://recursiveacm.devfolio.co",
        "schedule_url": "https://recursiveacm.devfolio.co/schedule",
        "status": "Active",
        "team_size": "2–4 members",
        "timeline": [
            "06 Sep 2026: Registrations begin",
            "25 Sep 2026: Extended PPT Submission deadline",
            "08 Oct 2026: Hackathon starts",
        ],
        "announcements": ["PPT submission deadline extended to 25 Sep 2026!"],
    }
    website_data = {
        "url": "https://recursiveacm.in",
        "venue": "Guru Nanak Institute of Technology, Kolkata",
        "tracks": ["AI & Intelligent Systems", "Web3"],
        "announcements": [],
    }

    md = syncer.compile_markdown(devfolio_data, website_data)
    assert "25 Sep 2026: Extended PPT Submission deadline" in md
    assert "PPT submission deadline extended" in md
    assert "https://recursiveacm.devfolio.co" in md
    assert "https://recursiveacm.in" in md


def test_live_sync_writes_file_and_calls_indexer(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    mock_indexer = MagicMock()
    mock_retriever = MagicMock()

    syncer = LiveWebSync(
        knowledge_dir=knowledge_dir,
        indexer=mock_indexer,
        retriever=mock_retriever,
    )

    fake_devfolio = {
        "name": "Recursive",
        "url": "https://recursiveacm.devfolio.co",
        "schedule_url": "https://recursiveacm.devfolio.co/schedule",
        "status": "Active",
        "team_size": "2–4 members",
        "timeline": ["20 Sep 2026: Registrations end"],
        "announcements": [],
    }
    fake_website = {
        "url": "https://recursiveacm.in",
        "venue": "GNIT Kolkata",
        "tracks": ["AI"],
        "announcements": [],
    }

    with patch.object(syncer, "fetch_devfolio", return_value=fake_devfolio), \
         patch.object(syncer, "fetch_website", return_value=fake_website):
        updated, content = syncer.sync(force=True)
        assert updated is True
        assert (knowledge_dir / "live_updates.md").exists()
        assert "20 Sep 2026" in content
        mock_indexer.build_index.assert_called_once()
        mock_retriever.load.assert_called_once()


@pytest.mark.asyncio
async def test_generator_attaches_live_context_on_deadline_query(tmp_path):
    mock_provider = MockProvider()
    mock_provider.answer = AsyncMock(return_value="The PPT submission deadline has been extended on Devfolio to September 25, 2026.")

    mock_live_sync = MagicMock()
    mock_live_sync.get_live_context.return_value = (
        "Devfolio Schedule:\n- 25 Sep 2026: Extended PPT idea submission deadline"
    )

    generator = AnswerGenerator(
        llm_provider=mock_provider,
        default_organizer_channel="#help",
        live_sync=mock_live_sync,
    )

    chunk = DocumentChunk(
        chunk_id="submission.md#1",
        source="submission.md",
        section="Deadlines",
        updated_at="2026-09-19",
        text="Submit your PPT on Devfolio.",
    )
    result = RetrievalResult(chunk=chunk, score=0.85)

    answer, was_fallback = await generator.generate_answer(
        question="Is there any chance the PPT submission deadline can be extended?",
        retrieval_results=[result],
    )

    assert was_fallback is False
    assert "extended on Devfolio" in answer
    mock_live_sync.get_live_context.assert_called()
    call_kwargs = mock_provider.answer.call_args.kwargs
    assert "Live Status from Devfolio" in call_kwargs["context"]
