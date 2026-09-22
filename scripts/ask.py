"""Interactive CLI test tool to query the Hackathon AI Agent.

Usage:
    python scripts/ask.py "Can a team have 5 members?"
    python scripts/ask.py "Where do we submit our project and what is the PPT format?"
    python scripts/ask.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai.classifier import MessageClassifier
from ai.embeddings import get_embedding_provider
from ai.generator import AnswerGenerator
from ai.provider import get_llm_provider
from config import config
from rag.retriever import KnowledgeRetriever
from storage.database import Database
from storage.memory import ConversationMemory
from storage.memory_store import MemoryStore

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")


async def ask(question: str) -> None:
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # 1. Initialize components
    db = Database(config.database_path)
    embedding_provider = get_embedding_provider(config)
    retriever = KnowledgeRetriever(
        faiss_index_path=config.faiss_index_path,
        metadata_path=config.metadata_path,
        embedding_provider=embedding_provider,
    )

    if not retriever.is_ready():
        print("[ERROR] FAISS index is not ready. Run `python scripts/rebuild_index.py` first.")
        return

    llm_provider = get_llm_provider(config)
    classifier = MessageClassifier(llm_provider=llm_provider)
    generator = AnswerGenerator(
        llm_provider=llm_provider,
        default_organizer_channel=config.organizer_channel_name,
        classifier=classifier,
        memory_store=MemoryStore(config.knowledge_dir),
    )
    memory = ConversationMemory(db=db, max_history_turns=6)

    print(f"[AI Provider]  {type(llm_provider).__name__} (Model: {config.llm_model})")

    # 2. Decision Layer
    is_mentioned = any(m in question.lower() for m in ["@bot", "@hackbot", "@recur"])
    should_reply, reason = await classifier.should_reply(
        content=question,
        is_bot_mentioned=is_mentioned,
        is_reply_to_bot=False,
    )
    print(f"[Decision]     Should Reply: {should_reply} | Reason: {reason}")

    if not should_reply:
        print("\n[BOT ACTION] Stayed silent (Message was classified as casual chatter).")
        print("=" * 70 + "\n")
        return

    # 3. Retrieve relevant chunks
    results = retriever.retrieve(question, top_k=4)
    print(f"[Retrieval]    Found {len(results)} relevant chunks in knowledge base:")
    for i, res in enumerate(results, 1):
        print(f"   {i}. [{res.chunk.source}] {res.chunk.section} (Score: {res.score:.3f})")

    # 4. Generate grounded answer
    history = memory.get_history_context(channel_id="cli_test", user_id="cli_user")
    organizer_tag_str = "@Core Member or @Volunteer"
    answer, was_fallback = await generator.generate_answer(
        question=question,
        retrieval_results=results,
        history=history,
        organizer_channel=config.organizer_channel_name,
        organizer_tag=organizer_tag_str,
    )

    print("\n" + "-" * 70)
    print("BOT ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    if "only related to this hackathon" in answer.lower():
        status = "OFF-TOPIC QUERY HANDLED (REFUSAL)"
    elif was_fallback:
        status = "SAFE FALLBACK TRIGGERED"
    else:
        status = "SUCCESSFULLY GROUNDED IN KNOWLEDGE"
    print(f"[Status]       {status}")
    print("=" * 70 + "\n")


async def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        await ask(query)
    else:
        # Run test battery of questions
        test_questions = [
            "What is the team size and can I apply solo?",
            "What are the 6 tracks we can build in?",
            "What is the story of the plastic chair?",
            "Where is the venue and how do I get there from Sodepur station?",
            "What is the slide structure for the idea PPT submission?",
            "Can we bring a live pet monkey to GNIT?",
        ]
        for q in test_questions:
            await ask(q)


if __name__ == "__main__":
    asyncio.run(main())
