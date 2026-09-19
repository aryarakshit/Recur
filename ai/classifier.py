"""Reply Decision Layer & Message Classifier.

Implements the two-stage reply decision pipeline:
1. Fast heuristic checks (keywords, patterns, greetings, emojis, noise)
2. LLM classifier for ambiguous messages
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ai.provider import LLMProvider

logger = logging.getLogger(__name__)

# Common hackathon terms from specification
HACKATHON_KEYWORDS = {
    # Core event
    "hackathon", "hack", "registration", "register", "registered", "registering",
    "deadline", "submission", "submit", "submitting", "submitted",
    "team", "teams", "teammate", "teammates", "solo", "member", "members",
    "eligibility", "eligible", "prize", "prizes", "award", "awards",
    "bounty", "bounties", "judging", "judge", "judges", "mentor", "mentors",
    "venue", "schedule", "timeline", "problem statement", "rules", "rule",
    "certificate", "certificates", "api", "apis", "project", "demo",
    "presentation", "pitch", "devpost", "allowed", "international",
    "student", "students", "recursive", "recur", "gnit", "sodepur", "devfolio",
    "food", "wifi", "sleep", "sleeping", "hardware", "travel", "offline", "online",
    "ppt", "slides", "website", "site", "link", "links", "url", "portal",
    "apply", "applying", "template", "form", "docs", "github", "discord",
    # Timing & Dates
    "start", "starts", "starting", "begin", "begins", "end", "ends",
    "timing", "timings", "duration", "hours", "october",
    # Cost & Fees
    "fee", "fees", "free", "cost", "pay", "payment", "charge", "charges", "price",
    # Location & Transit
    "location", "address", "reach", "campus", "college", "transport", "transit",
    "station", "route", "bus", "auto", "train",
    # Tracks & Topics
    "track", "tracks", "theme", "themes", "topic", "topics", "domain", "domains",
    # Swag & Amenities
    "swag", "swags", "goodie", "goodies", "tshirt", "t-shirt", "merch", "lunch", "dinner",
    "breakfast", "meal", "meals", "refreshment", "refreshments", "water", "snacks",
    # Participant & Attendance
    "attend", "attendance", "participate", "participation", "participant", "participants",
    "laptop", "charger", "bring", "requirement", "requirements",
    # Questions & Help
    "help", "doubt", "doubts", "query", "queries", "question", "questions", "info", "information",
    "details", "guidelines", "criteria",
}

# Casual chatter patterns to ignore
NOISE_PATTERNS = [
    r"^(\s*bro\s*|\s*dude\s*|\s*guys\s*|\s*yo\s*)*\s*(lol|lmao|haha|rofl|kek|xd)+\s*$",
    r"^(hi|hello|hey|sup|gm|gn|good\s+morning|good\s+night|good\s+evening)\s*(!+|\.+)*$",
    r"^(nice|cool|awesome|great|congrats|gg|rip|wow|super|agree|true|fr|ikr)\s*(!+|\.+)*$",
    r"^<a?:[a-zA-Z0-9_]+:[0-9]+>$",  # Discord custom emoji
    r"^[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\s]+$",  # Emoji-only strings
    r"^(ok|okay|k|np|ty|thanks|thank you|welcome|sure|yep|nope|yes|no)\s*(!+|\.+)*$",
]


class MessageClassifier:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    def is_chatter(self, text: str) -> bool:
        """Determines if the message is purely conversational noise/banter."""
        clean = text.strip().lower()
        if not clean:
            return True

        # Check noise regexes
        for pat in NOISE_PATTERNS:
            if re.match(pat, clean, re.IGNORECASE):
                return True

        # Check for message purely pinging another user with banter (e.g. "@Rahul check this", "look at this 😂")
        if re.match(r"^<@!?[0-9]+>\s+(look\s+at\s+this|check\s+this|see\s+this|lol|haha)\b", clean):
            return True

        return False

    def is_teammate_search(self, text: str) -> bool:
        """Detects peer-to-peer teammate recruitment, finding teammates, and LFG messages."""
        clean = text.strip().lower()

        # Questions about official rules or limits are NOT teammate searches
        # e.g. "What is the team size limit?", "Can I participate solo?"
        if any(clean.startswith(q) for q in ["what", "can", "is", "are", "how", "where"]):
            if any(term in clean for term in ["limit", "maximum", "rule", "rules", "allowed", "allow", "size", "solo", "minimum"]):
                return False

        patterns = [
            # Looking for members / teammates
            r"\blooking\s+for\s+.*(member|members|teammate|teammates|dev|developer|partner|team|group|person|people)\b",
            # Looking to join a team
            r"\blooking\s+to\s+join\s+.*team\b",
            r"\blooking\s+to\s+team\s*up\b",
            r"\blooking\s+for\s+(a\s+)?team\b",
            # Need member / teammates
            r"\bneed\s+.*(member|members|teammate|teammates|frontend|backend|dev|designer)\b",
            r"\b(team\s+needs?|team\s+requires?)\b",
            # Join my / our team
            r"\bjoin\s+(my|our|a)\s+team\b",
            # Interested kindly reply / dm / pm
            r"\b(interested\s+.*(dm|pm|reply|ping)|reply\s+or\s+dm|dm\s+me|pm\s+me|ping\s+me|contact\s+me)\b",
            # Team formation inquiries to peers: anyone want to join / team up
            r"\banyone\s+(want|wanna|interested)\s+(to\s+)?(join|team\s*up|partner)\b",
            r"\banyone\s+(need|needs|looking\s+for)\s+(a\s+)?(teammate|member|team)\b",
            r"\b(who\s+(wants?|wanna)\s+to\s+(join|team\s*up|partner))\b",
            r"\b(team\s*up\s+with(\s+me)?)\b",
            r"\b(forming\s+(a\s+)?team|building\s+(a\s+)?team|create\s+(a\s+)?team)\b",
            # LFG / spots left / vacancies
            r"\b(lfg|spots?\s+(available|left|open)|slots?\s+(available|left|open)|vacanc(y|ies)|openings?)\b",
            # Domains / roles available for team recruitment
            r"\bdomains?\s*[-:]\s*.*(ai|frontend|backend|web|app)\b",
            r"\bavailable\s+.*(domain|domains|role|roles|slot|slots|spot|spots)\b",
        ]
        for pat in patterns:
            if re.search(pat, clean):
                return True

        # Check combination of peer recruitment signals
        recruiting_signals = 0
        if any(w in clean for w in ["looking for", "need", "require", "searching for", "join"]):
            recruiting_signals += 1
        if any(w in clean for w in ["team", "teammate", "teammates", "member", "members"]):
            recruiting_signals += 1
        if any(w in clean for w in ["dm", "pm", "reply", "interested", "available", "domains", "frontend", "backend"]):
            recruiting_signals += 1

        if recruiting_signals >= 3:
            return True

        return False

    def is_addressed_to_human(self, text: str) -> bool:
        """Detects if message is addressed specifically to human mentors, judges, or organizers.

        Examples:
        - "Mentors, as today is the last day of submission..."
        - "Mentors: can you please check our project?"
        - "Hey mentors, are you available?"
        - "Hi mentor, quick question about our circuit diagram"
        - "Judges, will we be presenting on our laptops?"
        - "Core team, is food provided tonight?"
        - "Organizers, where can we get the WiFi credentials?"
        - "Can any mentor check our backend repo?"
        """
        clean = text.strip().lower()
        if not clean:
            return False

        # 1. Starting with greeting + staff role e.g. "hey mentors", "hi mentor", "hello judges", "dear organizers"
        if re.search(
            r"^\s*(?:hey|hi|hello|dear)\s+(?:mentors?|judges?|organizers?|core\s*(?:team|members?)|volunteers?|admins?|mods?|moderators?|sir|ma'?am)\b",
            clean,
        ):
            return True

        # 2. Starting with staff role directly addressed with punctuation or addressing pronouns:
        # e.g. "Mentors, as today is...", "Mentors: can we...", "Mentors please...", "Judges, ..."
        if re.search(
            r"^\s*(?:mentors?|judges?|organizers?|core\s*(?:team|members?)|volunteers?|admins?|sir|ma'?am)\s*[,:]",
            clean,
        ):
            return True

        if re.search(
            r"^\s*(?:mentors?|judges?|organizers?)\s+(?:please|kindly|can|could|would|as\b|we\b|i\b|our\b|my\b)",
            clean,
        ):
            return True

        # 3. Direct request to mentors/judges in chat:
        # e.g. "can any mentor help", "could a mentor review", "anyone from core team"
        if re.search(
            r"\b(?:can|could|would)\s+(?:any\s+)?(?:mentor|mentors|judge|judges|organizer|organizers)\s+(?:help|assist|check|review|guide|clarify|answer|tell|look)\b",
            clean,
        ):
            return True

        if re.search(r"\b(?:anyone\s+from\s+(?:the\s+)?(?:core\s*team|mentors|organizers|judges))\b", clean):
            return True

        return False

    def evaluate_heuristics(self, text: str) -> bool | None:
        """Evaluates heuristic rules.

        Returns:
            True: Definite hackathon question
            False: Definite chatter / unrelated / teammate search / addressed to human staff
            None: Ambiguous (defer to LLM classifier to understand what participant wants)
        """
        clean = text.strip().lower()

        # 1. Definite chatter check
        if self.is_chatter(clean):
            return False

        # 2. Definite teammate search / peer recruiting check
        if self.is_teammate_search(clean):
            return False

        # 3. Definite address to human mentors/judges/staff
        if self.is_addressed_to_human(clean):
            return False


        # Identity questions directed at the bot
        if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|who\s+is\s+recur|what\s+is\s+recur|tell\s+me\s+about\s+yourself|introduce\s+yourself)\b", clean):
            return True

        # Check explicit resource/link request phrases
        if any(phrase in clean for phrase in ["website link", "site link", "official website", "hackathon link", "registration link", "apply link", "template link", "ppt link", "slides link", "discord link"]):
            return True

        # Extract words and check keyword overlap
        words = set(re.findall(r"\b[a-z0-9_]+\b", clean))
        keyword_hits = words.intersection(HACKATHON_KEYWORDS)

        # Multi-word keywords
        if "problem statement" in clean:
            keyword_hits.add("problem statement")

        is_question = "?" in clean or any(clean.startswith(w) for w in [
            "what", "when", "where", "how", "can", "is", "are", "who", "which",
            "give", "send", "share", "provide", "tell", "will", "does", "do",
            "should", "could", "may"
        ])

        # Strong signal: has hackathon keywords AND formatted as an inquiry/question
        if keyword_hits and is_question:
            return True

        # Explicit short query phrases even without question mark (e.g., "submission deadline", "team size limit")
        if len(words) <= 5 and any(phrase in clean for phrase in [
            "submission deadline", "team size", "team limit", "registration fee",
            "registration deadline", "wifi password", "venue address", "presentation template"
        ]):
            return True

        # If very short message with 0 keywords, clearly chatter ONLY IF not formatted as a question
        if len(words) <= 4 and not keyword_hits and not is_question:
            return False

        # Otherwise ambiguous: message might be a natural language question phrased uniquely,
        # or a participant statement. Defer to LLM to understand what the participant wants!
        return None

    async def should_reply(
        self,
        content: str,
        is_bot_mentioned: bool,
        is_reply_to_bot: bool,
        context: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Main reply decision pipeline matching Section 5 specifications.

        Returns:
            (should_reply: bool, reason: str)
        """
        # Rule 1 & 2: Direct mention or reply to bot -> Always answer
        if is_bot_mentioned:
            return True, "Bot directly mentioned"

        if is_reply_to_bot:
            return True, "Reply to bot message"

        # Check if message is addressed specifically to human mentors/staff (stay quiet!)
        if self.is_addressed_to_human(content):
            return False, "Message addressed to human mentors/staff"

        # Rule 3: Fast heuristic check
        heuristic_result = self.evaluate_heuristics(content)
        if heuristic_result is True:
            return True, "Hackathon question identified by heuristic"
        if heuristic_result is False:
            return False, "Filtered out by noise/heuristic filter"


        # Rule 4: Ambiguous message -> Query LLM classifier
        logger.info("Message is ambiguous ('%s'). Calling LLM classifier.", content)
        try:
            res = await self.llm_provider.classify(message=content, context=context)
            should = bool(res.get("should_reply", False))
            confidence = float(res.get("confidence", 0.0))
            reason = res.get("reason", "LLM classification")
            logger.info("LLM Classifier result: should_reply=%s, confidence=%.2f, reason=%s", should, confidence, reason)
            return should, f"LLM Classifier ({reason})"
        except Exception as e:
            logger.error("Error in LLM classifier: %s", e)
            return False, f"LLM Classifier error: {e}"

    async def is_hackathon_related(self, content: str, context: Optional[str] = None) -> bool:
        """Determines if a question or message is genuinely related to this hackathon.

        Returns True if related to the hackathon; False if off-topic, general knowledge,
        chatter, or unrelated banter.
        """
        clean = content.strip().lower()
        if not clean or self.is_chatter(clean):
            return False

        # Fast heuristic check
        h = self.evaluate_heuristics(clean)
        if h is True:
            return True
        if h is False:
            return False

        # Ambiguous message: query LLM classifier
        try:
            res = await self.llm_provider.classify(message=clean, context=context)
            return bool(res.get("should_reply", False))
        except Exception as e:
            logger.error("Error evaluating is_hackathon_related: %s", e)
            return False
