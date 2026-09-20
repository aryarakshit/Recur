"""Answer Generation Layer.

Constructs grounded prompts, calls the LLM provider, and enforces the safe organizer
fallback policy whenever the official knowledge base lacks the requested information.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

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


def clean_cognitive_response(raw_text: str, question: str = "") -> tuple[str, str]:
    """Separates internal cognitive deliberation (Read, Understand, Think) from the user-facing reply.

    Returns:
        (clean_reply, reasoning_process)
    """
    cleaned = raw_text.strip()
    reasoning = ""

    # 1. XML tag extraction: <thinking>...</thinking> or <thought>...</thought>
    thinking_match = re.search(r"<(?:thinking|thought)>(.*?)</(?:thinking|thought)>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if thinking_match:
        reasoning = thinking_match.group(1).strip()
        cleaned = re.sub(r"<(?:thinking|thought)>.*?</(?:thinking|thought)>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Section header extraction: [READ] ... [UNDERSTAND] ... [THINK & DELIBERATE] ... [REPLY]
    reply_match = re.search(
        r"(?:^|\n)\s*(?:\[\s*(?:REPLY|FINAL ANSWER|RESPONSE)\s*\]|4\.\s*\[\s*REPLY\s*\]:?)\s*\n?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if reply_match:
        pre_reasoning = cleaned[:reply_match.start()].strip()
        post_answer = cleaned[reply_match.end():].strip()
        if post_answer:
            reasoning = (reasoning + "\n" + pre_reasoning).strip() if reasoning else pre_reasoning
            cleaned = post_answer

    # 3. Strip unwanted trailing source footnotes (e.g. "\n\nSource: ...")
    cleaned = re.sub(r"\n+(?:\*\*|__)?Sources?(?:\*\*|__)?\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    # 4. Filter repetitive greetings ("Hey there!", "Hi there! 👋") unless user explicitly greeted first
    if question:
        user_greeted = any(
            re.search(rf"\b{g}\b", question.lower())
            for g in ["hi", "hello", "hey", "sup", "greetings", "good morning", "good evening", "namaste"]
        )
        if not user_greeted:
            cleaned = re.sub(
                r"^(?:(?:hi|hey|hello)(?:\s+there|\s+team|\s+everyone|\s+folks)?\s*[!,\.]*\s*(?:[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf])*\s*\n*)+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

    if reasoning and question:
        logger.info("Cognitive Deliberation for '%s':\n%s", question[:60], reasoning)

    return cleaned, reasoning


class AnswerGenerator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        default_organizer_channel: str = "#help",
        classifier: MessageClassifier | None = None,
        live_sync: Any | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.default_organizer_channel = default_organizer_channel
        self.classifier = classifier
        self.live_sync = live_sync

    async def generate_answer(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        history: str = "",
        organizer_channel: str | None = None,
        organizer_tag: str = "@Core Member or @Volunteer",
    ) -> tuple[str, bool]:
        """Generates an answer based strictly on retrieved knowledge and live web updates.

        Returns:
            tuple of (answer_text, was_fallback)
        """
        channel = organizer_channel or self.default_organizer_channel

        # Check identity questions
        clean_q = question.strip().lower()
        if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|who\s+is\s+recur|what\s+is\s+recur|tell\s+me\s+about\s+yourself)\b", clean_q):
            return IDENTITY_REPLY, False

        # Check if query is asking about deadlines, extensions, PPT submissions, or live updates
        is_time_or_deadline_query = any(
            w in clean_q for w in [
                "deadline", "extend", "extended", "extension", "date", "dates", "ppt", "submission",
                "submit", "schedule", "time", "timeline", "devfolio", "website", "update", "latest",
                "change", "changes", "close", "closing", "last date"
            ]
        )

        # 1. If retrieval yielded zero relevant chunks
        relevant_results = [r for r in retrieval_results if r.is_relevant]
        if not relevant_results:
            if self.classifier:
                is_related = await self.classifier.is_hackathon_related(question)
                if not is_related:
                    logger.info("Question '%s' is not related to hackathon.", question)
                    return OFF_TOPIC_REPLY, False

            # If candidates were retrieved (e.g. conversational/long phrasing where similarity
            # was slightly below strict threshold), utilize top candidates for situational grounding
            if retrieval_results:
                relevant_results = retrieval_results[:4]

        if not relevant_results:
            # If question might have answer on Devfolio or website, try live sync
            if self.live_sync and is_time_or_deadline_query:
                logger.info("Query '%s' triggered live web fetch from Devfolio and recursiveacm.in", question)
                live_text = self.live_sync.get_live_context(force=True)
                if live_text:
                    live_context = f"[Live Official Web Updates from Devfolio & recursiveacm.in]:\n{live_text}"
                    try:
                        answer = await self.llm_provider.answer(
                            question=question,
                            context=live_context,
                            history=history,
                            organizer_channel=channel,
                            organizer_tag=organizer_tag,
                        )
                        if not any(f in answer.lower() for f in ["couldn't find", "could not find"]):
                            return answer, False
                    except Exception as e:
                        logger.debug("Live context answer generation failed: %s", e)

            logger.info("No relevant context found in official KB for question: '%s'", question)
            return SAFE_FALLBACK_TEMPLATE.format(organizer_tag=organizer_tag, channel=channel), True

        # Format context
        context_parts = []
        for i, res in enumerate(relevant_results, 1):
            chunk = res.chunk
            header = f"[Source {i}: {chunk.source} | Section: {chunk.section}]"
            context_parts.append(f"{header}\n{chunk.text}")

        # If query asks about deadlines, extensions, or dates, append live web updates to context
        if self.live_sync and is_time_or_deadline_query:
            try:
                live_text = self.live_sync.get_live_context(force=False)
                if live_text:
                    context_parts.append(
                        f"[Live Status from Devfolio (https://recursiveacm.devfolio.co) & https://recursiveacm.in]:\n{live_text}"
                    )
            except Exception as e:
                logger.debug("Could not attach live context: %s", e)

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

            # Clean cognitive reasoning from answer and log thought process
            answer, _ = clean_cognitive_response(answer, question=question)

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
            
            if is_fallback:
                # If model couldn't find it and we haven't checked live web sources yet, try live web sync
                if self.live_sync and not any("[Live Status" in p for p in context_parts):
                    logger.info("Fallback triggered on '%s'. Checking live Devfolio & website updates...", question)
                    live_text = self.live_sync.get_live_context(force=True)
                    if live_text:
                        retry_context = f"{formatted_context}\n\n---\n\n[Live Status from Devfolio & recursiveacm.in]:\n{live_text}"
                        try:
                            retry_answer = await self.llm_provider.answer(
                                question=question,
                                context=retry_context,
                                history=history,
                                organizer_channel=channel,
                                organizer_tag=organizer_tag,
                            )
                            retry_answer, _ = clean_cognitive_response(retry_answer, question=question)
                            if not any(ind in retry_answer.lower() for ind in fallback_indicators):
                                return retry_answer, False
                        except Exception as e:
                            logger.debug("Retry answer error: %s", e)

                return SAFE_FALLBACK_TEMPLATE.format(organizer_tag=organizer_tag, channel=channel), True

            return answer, is_fallback

        except Exception as e:
            logger.error("Error generating answer: %s", e)
            return (
                "AI service is temporarily unavailable. Please contact a maintainer.",
                True,
            )
