"""Batch QA evaluation runner for Recur.
Tests 20 hackathon questions and 7 non-hackathon queries through the complete agent pipeline.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import config
from storage.database import Database
from ai.provider import get_llm_provider
from ai.classifier import MessageClassifier
from ai.generator import AnswerGenerator
from rag.retriever import KnowledgeRetriever
from ai.embeddings import get_embedding_provider


HACKATHON_QUERIES = [
    ("What is the team size limit for Recursive 2026?", "Team Size Limit"),
    ("Can I participate solo in the hackathon?", "Solo Participation"),
    ("Can students from different colleges form a team together?", "Cross-College Eligibility"),
    ("Is there any registration fee to participate?", "Registration Fee"),
    ("What are the rules for the Round 1 PPT presentation?", "Round 1 PPT Rules"),
    ("Do we need a working prototype for Round 1 submission?", "Working Prototype vs Idea"),
    ("If we submit our presentation at the last minute before deadline, will it affect our selection?", "Last-Minute Submission Anxiety"),
    ("What are the judging criteria and rubric for project evaluation?", "Judging Rubric"),
    ("When and where is the offline hackathon taking place?", "Date & Venue"),
    ("What is the schedule and timeline for the hackathon day on 8 October?", "Day-of Schedule"),
    ("Can we use generative AI tools like ChatGPT or GitHub Copilot?", "AI Tools Policy"),
    ("What hardware or personal items should we bring to the venue?", "What to Bring"),
    ("Will Wi-Fi and food be provided during the 8-hour sprint?", "Food & Wi-Fi Amenities"),
    ("What are the tracks or problem statements for the hackathon?", "Tracks & Themes"),
    ("so recur what is the pricepool?", "Dynamic Memory: Prize Pool"),
    ("Who should we contact if we face an emergency or dispute?", "Contacts & Support"),
    ("Can first-year undergraduate students participate?", "Student Eligibility"),
    ("What is the official Devfolio submission link?", "Devfolio Link"),
    ("What happens if one of our teammates drops out at the last moment?", "Teammate Dropout Dilemma"),
    ("What are the prizes and bounties for winning teams?", "Prizes & Recognition"),
]

NON_HACKATHON_QUERIES = [
    ("Can you write an essay about Shakespeare's Hamlet for my English homework?", "Off-Topic: Homework"),
    ("Give me a quick recipe for chocolate chip cookies.", "Off-Topic: Cooking Recipe"),
    ("Who directed the movie Interstellar and who starred in it?", "Off-Topic: Movie Trivia"),
    ("What is the weather forecast in Tokyo today?", "Off-Topic: Weather"),
    ("Write a Python script to reverse a linked list.", "Off-Topic: Leetcode / Generic Coding"),
    ("Who won the FIFA World Cup in 2022?", "Off-Topic: Sports Trivia"),
    ("bro haha lol", "Noise: Casual Chatter"),
]


async def run_evaluation():
    print("Starting Recur Comprehensive Batch Evaluation...")
    db = Database(config.database_path)
    emb = get_embedding_provider(config)
    retriever = KnowledgeRetriever(config.faiss_index_path, config.metadata_path, emb)
    llm = get_llm_provider(config)
    classifier = MessageClassifier(llm_provider=llm)
    generator = AnswerGenerator(llm_provider=llm, classifier=classifier)

    results = []

    all_queries = [(q, cat, "Hackathon") for q, cat in HACKATHON_QUERIES] + \
                  [(q, cat, "Non-Hackathon") for q, cat in NON_HACKATHON_QUERIES]

    for idx, (query, subcat, cat_type) in enumerate(all_queries, 1):
        t0 = time.time()
        # 1. Classification check
        is_chatter = classifier.is_chatter(query)
        should_reply, reply_reason = await classifier.should_reply(
            content=query,
            is_bot_mentioned=True, # Direct mention mode to evaluate answers
            is_reply_to_bot=False,
        )

        # 2. Retrieval
        retrieval_results = retriever.retrieve(query=query, top_k=4)
        top_source = retrieval_results[0].chunk.source if retrieval_results else "None"
        top_score = round(retrieval_results[0].score, 3) if retrieval_results else 0.0

        # 3. Generation
        answer, was_fallback = await generator.generate_answer(
            question=query,
            retrieval_results=retrieval_results,
            organizer_channel="#ask-mentors",
            organizer_tag="@Core Member or @Volunteer",
        )
        latency = round(time.time() - t0, 2)

        record = {
            "index": idx,
            "query": query,
            "subcategory": subcat,
            "type": cat_type,
            "is_chatter": is_chatter,
            "should_reply": should_reply,
            "top_source": f"{top_source} ({top_score})",
            "answer": answer.strip(),
            "was_fallback": was_fallback,
            "latency": latency,
        }
        results.append(record)
        print(f"[{idx}/27] ({cat_type}) {query[:45]}... -> Latency: {latency}s | Ans: {answer[:60]}...")

    with open("scratch/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nEvaluation complete! Saved to scratch/eval_results.json")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
