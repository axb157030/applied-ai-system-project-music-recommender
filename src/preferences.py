"""
Validation of preferences that came from an LLM.

Gemini's JSON is untrusted input, so every value is checked here before it can
reach the recommender. This file does not call Gemini, so it is fully testable offline.
Field names match UserProfile in recommender.py.
"""

import json
import re
from collections import Counter
from difflib import SequenceMatcher

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
NOT_A_TASTE = {"allow_explicit", "only_languages"}

STRING_KEYS = {"favorite_genre", "favorite_mood", "language"}
UNIT_FLOAT_KEYS = {          # must be between 0 and 1
    "target_energy", "target_acousticness", "target_valence",
    "target_danceability", "target_instrumentalness",
}
LIST_KEYS = {"preferred_tags", "preferred_occasions", "only_languages"}
# Songs the user names ("more like Hotaru no Hikari"). They are free text: find_song() later
# matches each one to a catalog song, and the interview turns the match into ordinary preferences.
SONG_KEYS = {"liked_songs", "disliked_songs"}
MAX_SONG_MENTIONS = 5
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
        "only_languages": {str(s["language"]).lower() for s in songs if s.get("language")} - {"none"},
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

        elif key in SONG_KEYS:
            if isinstance(value, list):
                titles = [v.strip()[:150] for v in value if isinstance(v, str) and v.strip()]
                if titles:
                    prefs[key] = titles[:MAX_SONG_MENTIONS]

        elif key == "vocals":
            if isinstance(value, str) and value.lower() in ("vocal", "instrumental"):
                prefs[key] = value.lower()

        elif key == "allow_explicit":
            if isinstance(value, bool):
                prefs[key] = value

        # any other key is ignored on purpose
    return prefs


# ---------------------------------------------------------------------------
# Songs the user names
# ---------------------------------------------------------------------------

def _words(text):
    """Lowercase words only: 'Sha-La-La (cover)' -> 'sha la la cover'."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def find_song(mention, songs, cutoff=0.6):
    """
    Match free text such as "Hotaru no Hikari by rainychy" to one catalog song, or None.

    Done in code, not by Gemini, so the answer is always a real catalog song. Each song is
    scored by how much of the text its title and artist account for, tolerating typos.
    """
    text = _words(mention)
    if not text:
        return None
    best, best_score = None, 0.0
    for song in songs:
        title, artist = _words(song.get("title", "")), _words(song.get("artist", ""))
        title_score = 1.0 if title and title in text else SequenceMatcher(None, text, title).ratio()
        # Typed titles are often shorter than the catalog's ("Hotaru no Hikari" vs a long title with tags).
        words = text.split()
        if len(words) >= 2 and all(w in title.split() for w in words):
            title_score = max(title_score, 0.9)
        artist_score = max((SequenceMatcher(None, w, a).ratio() for w in words for a in artist.split()), default=0)
        score = title_score + (0.15 if artist_score >= 0.8 else 0)
        if score > best_score:
            best, best_score = song, score
    return best if best_score >= cutoff else None


def songs_by_artist(mention, songs, cutoff=0.8):
    """
    Catalog songs whose artist name appears (typos allowed) in the text. Used when the title
    is not in the catalog but the artist is: "Sha La La by Rainychy" -> the Rainych songs.
    Only words of 4+ letters count, so "the" or "by" never match an artist.
    """
    words = [w for w in _words(mention).split() if len(w) >= 4]
    found = []
    for song in songs:
        artist_words = _words(song.get("artist", "")).split()
        if any(SequenceMatcher(None, w, a).ratio() >= cutoff for w in words for a in artist_words):
            found.append(song)
    return found


# Numeric preferences copied from a song the user names, as (profile key, song column).
SEED_NUMBERS = (
    ("target_energy", "energy"), ("target_valence", "valence"), ("target_acousticness", "acousticness"),
    ("target_danceability", "danceability"), ("target_tempo", "tempo_bpm"),
)


def seed_preferences(seed_songs):
    """
    Preferences that describe "songs like these": the most common mood and genre, the
    shared occasions and tags, and the average of the numeric features.
    Returns {} when there are no songs. Everything is read from catalog rows, so it is valid.
    """
    if not seed_songs:
        return {}
    prefs = {}
    for key, column in (("favorite_mood", "mood"), ("favorite_genre", "genre"), ("language", "language")):
        values = [str(s[column]).lower() for s in seed_songs if s.get(column) and str(s[column]).lower() != "none"]
        if values:
            prefs[key] = Counter(values).most_common(1)[0][0]
    for key, column in (("preferred_occasions", "occasion_tags"), ("preferred_tags", "mood_tags")):
        tags = Counter(t.strip().lower() for s in seed_songs for t in str(s.get(column, "")).split("|") if t.strip())
        if tags:
            prefs[key] = [tag for tag, _count in tags.most_common(3)]
    vocals = {s.get("vocals") for s in seed_songs}
    if len(vocals) == 1 and vocals <= {"vocal", "instrumental"}:
        prefs["vocals"] = vocals.pop()
    for key, column in SEED_NUMBERS:
        values = [float(s[column]) for s in seed_songs if s.get(column) not in (None, "")]
        if values:
            prefs[key] = round(sum(values) / len(values), 3)
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


