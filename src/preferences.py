"""
Validation of preferences that came from an LLM.

Gemini's JSON is untrusted input, so every value is checked here before it can
reach the recommender. This file does not call Gemini, so it is fully testable offline.
Field names match UserProfile in recommender.py.
"""

import json

# ---------------------------------------------------------------------------
# Field rules, copied from UserProfile. The model's JSON is untrusted input, so
# every value is checked against these before it reaches the recommender.
# ---------------------------------------------------------------------------

# No field is required. The recommender scores only the preferences it is given, so any
# MIN_PREFERENCES of them are enough to rank songs. These are the preferences the interview
# can ask about, most useful first (mood and occasion matter most when scoring).
ASK_ORDER = (
    "favorite_mood", "preferred_occasions", "target_energy", "favorite_genre", "vocals",
    "target_valence", "target_acousticness", "language", "preferred_decade",
    "target_popularity", "target_tempo", "target_danceability", "target_duration",
)
MIN_PREFERENCES = 2
# allow_explicit is a filter, not a taste, so on its own it does not describe what to recommend.
NOT_A_TASTE = {"allow_explicit"}

STRING_KEYS = {"favorite_genre", "favorite_mood", "language"}
UNIT_FLOAT_KEYS = {          # must be between 0 and 1
    "target_energy", "target_acousticness", "target_valence",
    "target_danceability", "target_instrumentalness",
}
LIST_KEYS = {"preferred_tags", "preferred_occasions"}
# target_popularity: 0-100 | target_tempo (BPM) and target_duration (seconds): > 0
# preferred_decade: int   | vocals: "vocal" or "instrumental" | allow_explicit: bool


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_vocab(songs):
    """
    The label words that really exist in the catalog, from a list of song dicts (load_songs).
    Returns {"favorite_genre": {...}, "favorite_mood": {...}, "preferred_occasions": {...},
    "preferred_tags": {...}}. Pass it to clean_preferences so invented labels are dropped.
    """
    def split(field):
        return {t.strip().lower() for s in songs for t in str(s.get(field, "")).split("|") if t.strip()}
    return {
        "favorite_genre": {str(s["genre"]).lower() for s in songs if s.get("genre")},
        "favorite_mood": {str(s["mood"]).lower() for s in songs if s.get("mood")},
        "preferred_occasions": split("occasion_tags"),
        "preferred_tags": split("mood_tags"),
    }


def clean_preferences(raw, vocab=None):
    """
    Keep only valid UserProfile fields from the model's dict. Unknown keys,
    nulls, wrong types and out-of-range numbers are dropped (not guessed), so
    a missing key always means "the user didn't say" or "the value was unusable".

    vocab (optional, from build_vocab): genre, mood, occasion and tag labels that are not in
    the catalog are dropped too, because a label no song has would match nothing.
    """
    if not isinstance(raw, dict):
        return {}
    if vocab is not None and not isinstance(vocab, dict):
        raise ValueError("vocab must be a dict or None")

    prefs = {}
    for key, value in raw.items():
        if value is None:
            continue

        if key in STRING_KEYS:
            if isinstance(value, str) and value.strip():
                label = value.strip().lower()
                if vocab is None or key not in vocab or label in vocab[key]:
                    prefs[key] = label

        elif key in UNIT_FLOAT_KEYS:
            num = _to_float(value)
            if num is not None and 0 <= num <= 1:
                prefs[key] = num

        elif key == "target_popularity":
            num = _to_float(value)
            if num is not None and 0 <= num <= 100:
                prefs[key] = num

        elif key in ("target_tempo", "target_duration"):
            num = _to_float(value)
            if num is not None and num > 0:
                prefs[key] = num

        elif key == "preferred_decade":
            num = _to_float(value)
            if num is not None:
                prefs[key] = int(num)

        elif key in LIST_KEYS:
            if isinstance(value, list):
                items = [v.strip().lower() for v in value if isinstance(v, str) and v.strip()]
                if vocab is not None and key in vocab:
                    items = [v for v in items if v in vocab[key]]
                if items:
                    prefs[key] = items

        elif key == "vocals":
            if isinstance(value, str) and value.lower() in ("vocal", "instrumental"):
                prefs[key] = value.lower()

        elif key == "allow_explicit":
            if isinstance(value, bool):
                prefs[key] = value

        # any other key is ignored on purpose
    return prefs


def enough_to_recommend(prefs, minimum=MIN_PREFERENCES):
    """True when the user has given at least `minimum` preferences (any ones). allow_explicit does not count."""
    return sum(1 for key in prefs if key not in NOT_A_TASTE) >= minimum


def unknown_attributes(prefs):
    """Preferences the interview could still ask about, most useful first."""
    return [key for key in ASK_ORDER if key not in prefs]


def parse_preferences_json(text, vocab=None):
    """Turn the model's reply text into a cleaned preferences dict ({} if it is not usable JSON)."""
    text = (text or "").strip()
    # Models sometimes wrap JSON in ```json fences even when told not to.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return clean_preferences(raw, vocab)


