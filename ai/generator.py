"""Answer Generation Layer.

Constructs grounded prompts, calls the LLM provider, and enforces the safe organizer
fallback policy whenever the official knowledge base lacks the requested information.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.classifier import MessageClassifier
    from ai.provider import LLMProvider
    from rag.models import RetrievalResult

logger = logging.getLogger(__name__)

SAFE_FALLBACK_TEMPLATE = (
    "I couldn't find this information in the official hackathon knowledge base. "
    "Please tag {organizer_tag} for clarification."
)

OFF_TOPIC_REPLY = "Please ask me questions only related to this hackathon."
IDENTITY_REPLY = "I am a bot for helping and providing any info about the hackathon."


class AnswerGenerator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        default_organizer_channel: str = "#help",
        classifier: MessageClassifier | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.default_organizer_channel = default_organizer_channel
        self.classifier = classifier

    async def generate_answer(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        history: str = "",
        organizer_channel: str | None = None,
        organizer_tag: str = "@Core Member or @Volunteer",
    ) -> tuple[str, bool]:
        """Generates an answer based strictly on retrieved knowledge.

        Returns:
            tuple of (answer_text, was_fallback)
        """
        channel = organizer_channel or self.default_organizer_channel

        # Check identity questions
        clean_q = question.strip().lower()
        if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|who\s+is\s+recur|what\s+is\s+recur|tell\s+me\s+about\s+yourself)\b", clean_q):
            return IDENTITY_REPLY, False

        # 1. If retrieval yielded zero relevant chunks
        relevant_results = [r for r in retrieval_results if r.is_relevant]
        if not relevant_results:
            if self.classifier:
                is_related = await self.classifier.is_hackathon_related(question)
                if not is_related:
                    logger.info("Question '%s' is not related to hackathon.", question)
                    return OFF_TOPIC_REPLY, False

            logger.info("No relevant context found in official KB for question: '%s'", question)
            return SAFE_FALLBACK_TEMPLATE.format(organizer_tag=organizer_tag, channel=channel), True

        # Format context
        context_parts = []
        for i, res in enumerate(relevant_results, 1):
            chunk = res.chunk
            header = f"[Source {i}: {chunk.source} | Section: {chunk.section}]"
            context_parts.append(f"{header}\n{chunk.text}")
        formatted_context = "\n\n---\n\n".join(context_parts)

        # 2. Call LLM provider
        try:
            answer = await self.llm_provider.answer(
                question=question,
                context=formatted_context,
                history=history,
                organizer_channel=channel,
                organizer_tag=organizer_tag,
            )

            if "only related to this hackathon" in answer.lower():
                return OFF_TOPIC_REPLY, False

            # Check if model returned fallback indication
            fallback_indicators = [
                "couldn't find this information",
                "could not find this information",
                "couldn't find that in the official",
                "could not find that in the official",
                "not found in the official",
                "ask an organizer",
                "please ask an organizer",
                "tag a server maintainer",
                "tag a core member",
                "tag an organizer",
            ]
            is_fallback = any(indicator in answer.lower() for indicator in fallback_indicators)
            if not is_fallback:
                # Strip unwanted trailing source footnote (e.g. "\n\nSource: ...")
                answer = re.sub(r"\n+(?:\*\*|__)?Sources?(?:\*\*|__)?\s*:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
            return answer, is_fallback

        except Exception as e:
            logger.error("Error generating answer: %s", e)
            return (
                "AI service is temporarily unavailable. Please contact a maintainer.",
                True,
            )
