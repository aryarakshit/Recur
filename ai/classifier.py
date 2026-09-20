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
    "team", "teams", "teammate", "teammates", "teamate", "teamates", "solo", "member", "members",
    "finding", "seek", "seeking",
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
    r"^(hi|hello|hey|sup|gm|gn|good\s+morning|good\s+night|good\s+evening)(\s+(guys|all|everyone|folks|people|there|y'all))?\s*(!+|\.+)*$",
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
            # Direct "finding / find / seeking / looking for / need / want" teammates or team (including typos like 'teamates')
            r"\b(?:i\s*(?:'?m|am)?\s*)?(?:find|finding|search|searching|seek|seeking|look|looking|need|needing|want|wanting)\s+(?:for\s+)?(?:a\s+|an\s+|any\s+|some\s+|more\s+|new\s+|good\s+)?(?:team|teams|team-?m?ates?|members?|partners?|squad)\b",
            # Finding teammates with typo or short form
            r"\b(?:find|finding|look|looking|seek|seeking)\s+(?:a\s+)?(?:team|teams|teammates?|teamates?)\b",
            # Looking for members / teammates
            r"\blooking\s+for\s+.*(member|members|teammate|teammates|teamate|teamates|dev|developer|partner|team|group|person|people)\b",
            # Looking to join a team
            r"\blooking\s+to\s+join\s+.*team\b",
            r"\blooking\s+to\s+team\s*up\b",
            r"\blooking\s+for\s+(a\s+)?team\b",
            # Need member / teammates
            r"\bneed\s+.*(member|members|teammate|teammates|teamate|teamates|frontend|backend|dev|designer)\b",
            r"\b(team\s+needs?|team\s+requires?)\b",
            # Join my / our team
            r"\bjoin\s+(my|our|a)\s+team\b",
            # Interested kindly reply / dm / pm
            r"\b(interested\s+.*(dm|pm|reply|ping)|reply\s+or\s+dm|dm\s+me|pm\s+me|ping\s+me|contact\s+me)\b",
            # Team formation inquiries to peers: anyone want to join / team up
            r"\banyone\s+(want|wanna|interested)\s+(to\s+)?(join|team\s*up|partner)\b",
            r"\banyone\s+(need|needs|looking\s+for|finding|search|searching\s+for)\s+(a\s+)?(teammate|teammates|teamate|teamates|member|members|team)\b",
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
        if any(w in clean for w in ["looking for", "need", "require", "searching for", "searching", "join", "find", "finding", "seek", "seeking", "want"]):
            recruiting_signals += 1
        if any(w in clean for w in ["team", "teams", "teammate", "teammates", "teamate", "teamates", "member", "members", "partner", "partners", "squad"]):
            recruiting_signals += 1
        if any(w in clean for w in ["dm", "pm", "reply", "interested", "available", "domains", "frontend", "backend"]):
            recruiting_signals += 1

        if recruiting_signals >= 2 and any(w in clean for w in ["find", "finding", "seeking", "looking for", "searching for", "need"]):
            return True

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

    def is_peer_conversation(self, text: str) -> bool:
        """Detects if message is a peer-to-peer discussion, chatting with other hackers,
        or soliciting opinions from other participants, rather than asking the bot.
        """
        clean = text.strip().lower()
        if not clean:
            return False

        peer_patterns = [
            # Addressing the chat room / peers
            r"^\s*(?:hey|hi|hello|yo)?\s*(?:guys|everyone|folks|people|all|y'?all|buddies|friends)\b",
            # Peer opinion / experience inquiries
            r"\b(?:what\s+do\s+(?:you\s+guys|y'?all|you\s+all)\s+think)\b",
            r"\b(?:has\s+anyone\s+(?:tried|used|tested|worked\s+with|seen|done))\b",
            r"\b(?:is\s+anyone\s+else)\b",
            r"\b(?:anyone\s+(?:here\s+)?(?:using|building|submitting|facing|working|doing))\b",
            r"\b(?:which|what)\s+(?:tech\s+stack|framework|library|tools?)\s+are\s+(?:you|y'?all|you\s+guys)\s+using\b",
            r"\b(?:how\s+is\s+(?:everyone|everybody|y'?all|your\s+team)\s+(?:doing|going))\b",
            r"\b(?:anyone\s+want\s+to\s+(?:share|see|test))\b",
            r"\b(?:anyone\s+(?:wants?|wanna|want)\s+to\s+(?:play|game|hang|chill|watch|call))\b",
            r"\b(?:is\s+it\s+just\s+me\s+or)\b",
        ]
        for pat in peer_patterns:
            if re.search(pat, clean):
                return True
        return False

    def is_addressed_to_other_user(self, text: str) -> bool:
        """Detects if message is explicitly addressed to another user (e.g. '@someone ...' or '<@123> ...')."""
        clean = text.strip().lower()
        if not clean:
            return False
        # Matches mention tags like <@123456> or plain-text pings like @heyimsouvik at the beginning of message
        if re.match(r"^\s*(?:<@!?\d+>|@[a-zA-Z0-9_\.\-]+)\b", clean):
            return True
        return False

    def evaluate_heuristics(self, text: str) -> bool | None:
        """Evaluates heuristic rules.

        Returns:
            True: Definite hackathon question
            False: Definite chatter / unrelated / teammate search / addressed to human staff / peer chat
            None: Ambiguous / Situational (defer to LLM classifier to understand what participant wants)
        """
        clean = text.strip().lower()

        # 1. Definite chatter check
        if self.is_chatter(clean):
            return False

        # 2. Definite address to another user
        if self.is_addressed_to_other_user(clean):
            return False

        # 3. Definite teammate search / peer recruiting check
        if self.is_teammate_search(clean):
            return False

        # 4. Definite address to human mentors/judges/staff
        if self.is_addressed_to_human(clean):
            return False

        # 5. Definite peer-to-peer conversation among hackers
        if self.is_peer_conversation(clean):
            return False

        # 6. Definite author self-resolution / acknowledgment (e.g. "never mind", "got it thanks")
        if re.search(r"^\s*(?:never\s*mind|nevermind|nvm|got\s*it|all\s*good|all\s*clear|resolved)\b", clean):
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

        # If very short message with 0 keywords, clearly chatter ONLY IF not formatted as a question
        if len(words) <= 4 and not keyword_hits and not is_question:
            return False

        # Explicit short query phrases even without question mark (e.g., "submission deadline", "team size limit")
        if len(words) <= 5 and any(phrase in clean for phrase in [
            "submission deadline", "team size", "team limit", "registration fee",
            "registration deadline", "wifi password", "venue address", "presentation template"
        ]):
            return True

        # Direct, concise official hackathon question (short and focused on rules/facts)
        # e.g., "When is registration closing?", "Can international students participate?", "Is registration free?"
        is_conversational_narrative = any(phrase in clean for phrase in [
            "working on", "me and my team", "our team is", "still working", "trying to build",
            "what do you think", "what should we", "will that anyhow", "affect our"
        ])
        if keyword_hits and is_question and len(words) <= 12 and not is_conversational_narrative:
            return True

        # Otherwise ambiguous / situational: message might be a conversational dilemma,
        # natural language question, or participant situation. Defer to LLM to understand what the participant wants!
        return None
    def is_already_answered_in_context(self, context: str) -> bool:
        """Determines if the situational context shows that a mentor, staff, or peer already answered or resolved."""
        if not context:
            return False
        ctx = context.lower()
        ack_phrases = [
            "author acknowledged", "thank you", "thanks mentor", "got it thanks",
            "thank you!", "thanks!", "understood, thanks", "makes sense, thanks",
            "ok got it", "okay got it", "never mind", "nevermind", "all clear",
            "already answered", "mentor/staff", "handled by staff",
        ]
        return any(p in ctx for p in ack_phrases)

    async def should_reply(
        self,
        content: str,
        is_bot_mentioned: bool,
        is_reply_to_bot: bool,
        context: Optional[str] = None,
        has_other_mentions: bool = False,
        is_reply_to_other: bool = False,
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

        # Check if message is addressed to another user (via mention, reply, or @tag)
        if has_other_mentions or is_reply_to_other or self.is_addressed_to_other_user(content):
            return False, "Message addressed to another user"

        # Check if message is addressed specifically to human mentors/staff (stay quiet!)
        if self.is_addressed_to_human(content):
            return False, "Message addressed to human mentors/staff"

        # Check if message is peer-to-peer discussion among participants (stay quiet!)
        if self.is_peer_conversation(content):
            return False, "Peer-to-peer discussion / chat among participants"

        # Check situational context: If context clearly indicates already resolved or answered, stay silent!
        if context and self.is_already_answered_in_context(context):
            return False, "Already answered or resolved in channel context"

        # Rule 3: Fast heuristic check
        heuristic_result = self.evaluate_heuristics(content)
        if heuristic_result is False:
            return False, "Filtered out by noise/heuristic filter"

        # If heuristic is True AND there is no subsequent context to evaluate, reply immediately
        if heuristic_result is True and not context:
            return True, "Hackathon question identified by heuristic"

        # Rule 4: Ambiguous message OR message with subsequent channel context -> Query LLM classifier
        logger.info("Evaluating message with situational context ('%s'). Calling LLM classifier.", content)
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
