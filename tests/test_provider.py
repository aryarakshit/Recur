"""Unit tests for the LLM provider and generator layers."""

import pytest
from ai.generator import AnswerGenerator
from ai.provider import MockProvider, get_llm_provider
from config import Config
from rag.models import DocumentChunk, RetrievalResult


@pytest.mark.asyncio
async def test_mock_provider_classification():
    provider = MockProvider()
    res = await provider.classify("When is the registration deadline?")
    assert res["should_reply"] is True

    res2 = await provider.classify("hello there")
    assert res2["should_reply"] is False


@pytest.mark.asyncio
async def test_generator_safe_fallback_on_empty_retrieval():
    provider = MockProvider()
    generator = AnswerGenerator(llm_provider=provider, default_organizer_channel="#help")

    # Empty retrieval
    answer, was_fallback = await generator.generate_answer(
        question="Can we bring a pet llama?",
        retrieval_results=[],
        organizer_channel="#help",
    )

    assert was_fallback is True
    assert "couldn't find this information in the official hackathon knowledge base" in answer
    assert "@Core Member or @Volunteer" in answer


@pytest.mark.asyncio
async def test_generator_with_valid_context():
    provider = MockProvider()
    generator = AnswerGenerator(llm_provider=provider, default_organizer_channel="#help")

    chunk = DocumentChunk(
        chunk_id="test#1",
        source="rules.md",
        section="Team Limits",
        updated_at="2026-09-19",
        text="Maximum team size is 4 members.",
    )
    results = [RetrievalResult(chunk=chunk, score=0.88)]

    answer, was_fallback = await generator.generate_answer(
        question="What is team size?",
        retrieval_results=results,
        organizer_channel="#help",
    )

    assert was_fallback is False
    assert "Maximum team size is 4 members" in answer


def test_factory_returns_mock_when_keys_blank():
    cfg = Config()
    cfg.gemini_api_key = ""
    cfg.groq_api_key = ""

    provider = get_llm_provider(cfg)
    assert isinstance(provider, MockProvider)


@pytest.mark.asyncio
async def test_generator_refuses_off_topic_query():
    from ai.classifier import MessageClassifier

    provider = MockProvider()
    classifier = MessageClassifier(llm_provider=provider)
    generator = AnswerGenerator(llm_provider=provider, classifier=classifier)

    # Completely off-topic question with no relevant chunks
    answer, was_fallback = await generator.generate_answer(
        question="What is the weather today?",
        retrieval_results=[],
        organizer_channel="#help",
    )

    assert was_fallback is False
    assert "Please ask me questions only related to this hackathon" in answer


def test_clean_cognitive_response_section_headers():
    from ai.generator import clean_cognitive_response

    raw = """[READ]
User asks about submitting late.
[UNDERSTAND]
They are stressed about cutoff penalties.
[THINK & DELIBERATE]
Official policy has zero penalty before cutoff.
[REPLY]
Submitting at the last minute has zero negative impact on selection! Just be sure to submit before the cutoff."""

    answer, reasoning = clean_cognitive_response(raw)
    assert answer == "Submitting at the last minute has zero negative impact on selection! Just be sure to submit before the cutoff."
    assert "[READ]" in reasoning
    assert "[UNDERSTAND]" in reasoning


def test_clean_cognitive_response_xml_tags():
    from ai.generator import clean_cognitive_response

    raw = """<thinking>
User wants to know team sizes. Rule is 2-4 members.
</thinking>
Teams must have between 2 and 4 members!

Sources: rules.md"""

    answer, reasoning = clean_cognitive_response(raw)
    assert answer == "Teams must have between 2 and 4 members!"
    assert "User wants to know team sizes" in reasoning
    assert "<thinking>" not in answer
    assert "Sources:" not in answer


def test_clean_cognitive_response_strips_unnecessary_greeting():
    from ai.generator import clean_cognitive_response

    raw = """[REPLY]
Hey there! 👋

For RECURSIVE — Shift-8 Hackathon 2026, we are running an Open Theme competition."""

    # Direct query without greeting -> Greeting stripped
    ans, _ = clean_cognitive_response(raw, question="Can you share problem statements with me")
    assert ans == "For RECURSIVE — Shift-8 Hackathon 2026, we are running an Open Theme competition."
    assert "Hey there" not in ans

    # Query with greeting -> Greeting preserved
    ans2, _ = clean_cognitive_response(raw, question="Hi! Can you share problem statements with me")
    assert ans2.startswith("Hey there! 👋")


