"""Answer Generation Layer.

Constructs grounded prompts, calls the LLM provider, and enforces the safe organizer
fallback policy whenever the official knowledge base lacks the requested information.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, TYPE_CHECKING

from rag.retriever import normalize_query_text

if TYPE_CHECKING:
    from ai.classifier import MessageClassifier
    from ai.provider import LLMProvider
    from rag.models import RetrievalResult
    from storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)

SAFE_FALLBACK_TEMPLATE = (
    "I couldn't find this information in the official hackathon knowledge base. "
    "Please tag {organizer_tag} for clarification."
)

OFF_TOPIC_REPLY = "Please ask me questions only related to this hackathon."
IDENTITY_REPLY = "I am a bot for helping and providing any info about the hackathon."

# Questions about hard facts that organizers change on Devfolio or the website. These
# re-check the live sources before answering (LiveWebSync caps it at one fetch a minute).
HARD_INFO_RE = re.compile(
    r"\b(?:deadlines?|dates?|when|extend(?:ed|s)?|extensions?|schedule|timeline|timings?|time|"
    r"register(?:ed)?|registrations?|submi(?:t|ts|tted|tting|ssions?)|ppt|prizes?|pool|rewards?|cash|"
    r"results?|shortlist(?:ed)?|selected|selection|rounds?|venue|location|address|fees?|open|opens|"
    r"close[sd]?|closing|last\s+date|devfolio|website|latest|updates?|announce(?:d|ments?)?|status)\b",
    re.IGNORECASE,
)


def clean_cognitive_response(raw_text: str, question: str = "") -> tuple[str, str]:
    """Separates internal cognitive deliberation (Read, Understand, Think) from the user-facing reply.

    Returns:
        (clean_reply, reasoning_process)
    """
    cleaned = raw_text.strip()
    reasoning = ""

    # 1. XML tag extraction: <think>...</think> (Qwen), <thinking>...</thinking> or <thought>...</thought>
    thinking_match = re.search(r"<(think|thinking|thought)>(.*?)</\1>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if thinking_match:
        reasoning = thinking_match.group(2).strip()
        cleaned = re.sub(r"<(think|thinking|thought)>.*?</\1>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    # A reasoning block cut off by the token limit never closes; none of it is the answer.
    unclosed = re.search(r"<(?:think|thinking|thought)>", cleaned, flags=re.IGNORECASE)
    if unclosed:
        reasoning = (reasoning + "\n" + cleaned[unclosed.end():]).strip()
        cleaned = cleaned[:unclosed.start()].strip()

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
    elif re.search(r"\[\s*(?:READ|THINK|UNDERSTAND)\s*\]", cleaned, flags=re.IGNORECASE):
        reasoning = cleaned
        # Strip all [READ], [UNDERSTAND], and [THINK] blocks to avoid leaking internal deliberation
        cleaned = re.sub(r"\[\s*(?:READ|UNDERSTAND|THINK(?:\s*&\s*DELIBERATE)?)\s*\].*?(?=(?:\[\s*[A-Z]|\Z))", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

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
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.default_organizer_channel = default_organizer_channel
        self.classifier = classifier
        self.live_sync = live_sync
        self.memory_store = memory_store
        self._semaphore = asyncio.Semaphore(8)

    async def _live_context(self, refresh: bool, force: bool = False) -> str:
        """Live Devfolio/website status. Refreshing runs the scrape in a worker thread
        so the Discord event loop never blocks on the network."""
        if not self.live_sync:
            return ""
        try:
            if refresh or force:
                text = await asyncio.to_thread(self.live_sync.get_live_context, force)
            else:
                text = self.live_sync.cached_context()
        except Exception as e:
            logger.debug("Could not load live context: %s", e)
            return ""
        return text if isinstance(text, str) else ""

    async def generate_answer(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        history: str = "",
        organizer_channel: str | None = None,
        organizer_tag: str = "@Core Member or @Volunteer",
        channel_name: str | None = None,
    ) -> tuple[str, bool]:
        """Generates an answer from organizer memory, retrieved knowledge and live web updates.

        Returns:
            tuple of (answer_text, was_fallback)
        """
        channel = organizer_channel or self.default_organizer_channel

        # Check identity questions
        clean_q = question.strip().lower()
        if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|who\s+is\s+recur|what\s+is\s+recur|tell\s+me\s+about\s+yourself)\b", clean_q):
            return IDENTITY_REPLY, False

        # Check friendly greetings (e.g. "heloow recur", "hello recur", "hi recur", "hey recur", "gm", "good morning")
        if re.search(r"^(?:hi|hello|helo|helow|heloow|hey|sup|greetings|gm|good\s+(?:morning|afternoon|evening))\s*(?:recur|bot)?\s*[!.]*$", clean_q):
            return "Hello! I am Recur, the official AI assistant for RECURSIVE 2026. How can I help you with the hackathon today?", False

        # Check conversational check-ins (e.g. "how are you", "how are you doing", "what's up")
        if re.search(r"\b(?:how\s+are\s+you|how's\s+it\s+going|how\s+are\s+you\s+doing|what's\s+up|wassup)\b", clean_q):
            return "I'm doing well, thank you! Ready to assist you with any questions or guidelines for RECURSIVE 2026. What would you like to know?", False

        # Organizer memory from #recur-mem-update goes into every prompt, so a saved note
        # applies until an organizer deletes it, however the question is worded.
        memory_notes = self.memory_store.prompt_block() if self.memory_store else ""
        memory_hit = bool(memory_notes) and self.memory_store.is_relevant_to(normalize_query_text(question))

        relevant_results = [r for r in retrieval_results if r.is_relevant]
        if not relevant_results and not memory_hit:
            if self.classifier:
                is_related = await self.classifier.is_hackathon_related(question)
                if not is_related:
                    logger.info("Question '%s' is not related to hackathon.", question)
                    return OFF_TOPIC_REPLY, False

        # If candidates were retrieved (e.g. conversational/long phrasing where similarity
        # was slightly below strict threshold), utilize top candidates for situational grounding
        if not relevant_results:
            relevant_results = retrieval_results[:4]

        # Hard facts (dates, deadlines, prizes, registration...) are re-checked against
        # Devfolio and recursiveacm.in before answering; so is anything the KB can't answer.
        live_text = await self._live_context(refresh=bool(HARD_INFO_RE.search(clean_q)) or not relevant_results)

        if not (relevant_results or memory_notes or live_text):
            logger.info("No relevant context found in official KB for question: '%s'", question)
            return SAFE_FALLBACK_TEMPLATE.format(organizer_tag=organizer_tag, channel=channel), True

        context_parts = []
        if channel_name:
            context_parts.append(f"[Question asked in Discord channel: #{channel_name}]")
        if memory_notes:
            context_parts.append(
                "[Organizer Notes from #recur-mem-update — highest priority; if two notes conflict, the higher # wins]\n"
                + memory_notes
            )
        for i, res in enumerate(relevant_results, 1):
            chunk = res.chunk
            header = f"[Source {i}: {chunk.source} | Section: {chunk.section}]"
            context_parts.append(f"{header}\n{chunk.text}")
        if live_text:
            context_parts.append(
                f"[Live Status from Devfolio (https://recursiveacm.devfolio.co) & https://recursiveacm.in]:\n{live_text}"
            )

        formatted_context = "\n\n---\n\n".join(context_parts)

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

        # Call LLM provider with concurrency semaphore (max 8 parallel LLM calls)
        try:
            async with self._semaphore:
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

            is_fallback = not answer or any(indicator in answer.lower() for indicator in fallback_indicators)

            if is_fallback:
                # Live sources weren't available yet: fetch them now and try once more.
                if self.live_sync and not live_text:
                    logger.info("Fallback triggered on '%s'. Checking live Devfolio & website updates...", question)
                    live_text = await self._live_context(refresh=True, force=True)
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
                            if retry_answer and not any(ind in retry_answer.lower() for ind in fallback_indicators):
                                return retry_answer, False
                        except Exception as e:
                            logger.debug("Retry answer error: %s", e)

                return SAFE_FALLBACK_TEMPLATE.format(organizer_tag=organizer_tag, channel=channel), True

            return answer, False

        except Exception as e:
            logger.error("Error generating answer: %s", e)
            return (
                "AI service is temporarily unavailable. Please contact a maintainer.",
                True,
            )
