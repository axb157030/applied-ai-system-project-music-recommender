"""
Input guardrails for the music recommender.

WHY THIS IS NOT A LONG BLOCKLIST
    A word list cannot tell "I hate sad songs" from hate speech, or the song
    "Cocaine" from a drug request. In a music app those false alarms are the
    common case, and they block exactly the opinions the interview needs
    ("I hate loud music", "something dark", "protest songs").

    So this file only checks things that can actually hurt the user or the system:
      - self-harm intent    -> a caring reply, never a "not allowed" refusal
      - personal data       -> kept out of Gemini and out of the log files
      - prompt injection    -> stops users from hijacking the model
      - oversized input     -> cost and abuse limit

WHAT ALREADY PROTECTS THE APP BY DESIGN (no keywords needed)
      - preferences.py only accepts values from fixed fields and the catalog's own
        labels, so Gemini cannot put anything else into the profile
      - the recommender only returns songs that exist in the catalog
      - allow_explicit filters explicit songs out
      - the retriever refuses questions the docs cannot answer, and RAG answers may
        only use the retrieved paragraphs
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

MAX_INPUT_CHARS = 500

SELF_HARM_MESSAGE = (
    "I'm really sorry you're feeling this way, and I'm glad you said something. "
    "You don't have to go through it alone. If you're in the US, you can call or "
    "text 988 (Suicide & Crisis Lifeline) at any time. If you're somewhere else, "
    "please contact your local emergency number or a crisis line in your country. "
    "When you're ready, I'm happy to help you find some music."
)
INJECTION_MESSAGE = (
    "I can only help with finding music. Tell me what you'd like to listen to, "
    "or ask me about energy, tempo, moods and genres."
)
PERSONAL_DATA_MESSAGE = (
    "Please don't share personal details like emails, phone numbers or card "
    "numbers. I only need to know your music tastes."
)
TOO_LONG_MESSAGE = f"That's a bit long for me. Please keep it under {MAX_INPUT_CHARS} characters."
INVALID_MESSAGE = "I couldn't read that. Please type your answer again."


@dataclass(frozen=True)
class GuardrailResult:
    """allowed=False means: do not send the text to Gemini, show `message` instead."""
    allowed: bool
    category: Optional[str] = None   # which rule fired, for the log
    message: Optional[str] = None    # what to show the user


# (category, pattern, message), checked in order. Self-harm comes first so it is
# never hidden behind another category.
_RULES = (
    ("self_harm", re.compile(
        r"\b(?:(?:kill|hurt|harm|cut)(?:ing)?\s+myself"
        r"|end(?:ing)?\s+my\s+life"
        r"|take\s+my\s+own\s+life"
        r"|(?:want|wanna|going)\s+to\s+die"
        r"|don'?t\s+want\s+to\s+(?:live|be\s+here|exist)"
        r"|suicidal"
        r"|self[- ]?harm)\b", re.IGNORECASE), SELF_HARM_MESSAGE),
    ("prompt_injection", re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}\b(?:instructions?|prompts?|rules|guidelines)\b"
        r"|\b(?:system|developer|hidden)\s+prompt\b"
        r"|\byou\s+are\s+now\b"
        r"|\bpretend\s+(?:to\s+be|you(?:'re|\s+are))\b", re.IGNORECASE), INJECTION_MESSAGE),
    ("personal_data", re.compile(
        r"[\w.+-]+@[\w-]+\.[\w.-]+"                                           # email
        r"|(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"                                    # US SSN
        r"|(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"    # US phone
        , re.IGNORECASE), PERSONAL_DATA_MESSAGE),
)

_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,15}\d(?!\d)")
_INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")


def _normalize(text):
    """Fold look-alike characters and drop zero-width ones so they can't dodge the patterns."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _luhn_ok(digits):
    """Card-number checksum. Stops a list of years like '1990 2000 2010 2020' being flagged."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _has_card_number(text):
    return any(_luhn_ok(re.sub(r"\D", "", m.group())) for m in _CARD_CANDIDATE.finditer(text))


def check_input(text):
    """Check what the user typed. Returns a GuardrailResult."""
    if not isinstance(text, str):
        return GuardrailResult(False, "invalid", INVALID_MESSAGE)

    clean = _normalize(text)
    if len(clean) > MAX_INPUT_CHARS:
        return GuardrailResult(False, "too_long", TOO_LONG_MESSAGE)

    for category, pattern, message in _RULES:
        if pattern.search(clean):
            return GuardrailResult(False, category, message)

    if _has_card_number(clean):
        return GuardrailResult(False, "personal_data", PERSONAL_DATA_MESSAGE)

    return GuardrailResult(True)


def check_guardrails(text):
    """Kept for older callers: a list of violation messages (empty when the text is fine)."""
    result = check_input(text)
    return [] if result.allowed else [result.message]