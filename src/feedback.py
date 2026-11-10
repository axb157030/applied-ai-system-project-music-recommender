"""
Feedback: turn the user's 1-10 ratings (and optional comments) into
  (1) a measurement of how well the recommender did, and
  (2) a better profile for the next round.

Ratings are keyed by SONG ID ({7: 9, 16: 2, ...}), because titles are not always unique.

    validate_ratings   rejects bad ratings (0, 11, "ten", True, unknown song, unrated songs)
    compute_metrics    mean rating, hit rate (share rated 7+), liked / disliked song ids
    update_profile     moves numeric targets toward the best-rated songs and away from the worst
    learn_labels       learns mood, genre, occasions and tags from ALL ratings so far
    apply_comments     turns comments like "too slow" into profile changes (uses Gemini, checked)
    regenerate         runs all of the above and returns a fresh top-k of unseen songs

Why the numbers are NOT done by Gemini: moving a target toward the average of the songs the
user liked is arithmetic. Code does it exactly, for free, the same way every time, and it can
be unit tested. (A first version asked Gemini for the whole new profile. On the real API it
crashed on a ```json code fence and invented preferences the user never gave.)

Why comments ARE done by Gemini: "too slow" is language, and code cannot read language.
Gemini only proposes preferences. preferences.clean_preferences checks them before they
touch the profile, and if Gemini fails the ratings still work.
"""

from .guardrails import check_input
from .llm import GeminiClient, LLMError
from .preferences import clean_preferences
from .recommender import SONG_FIELD_FOR_FEATURE, recommend_songs

MIN_RATING = 1
MAX_RATING = 10
LIKE_AT_LEAST = 7       # rated 7 or more = the user liked it (counts toward hit rate)
DISLIKE_AT_MOST = 4     # rated 4 or less = the user disliked it
COMMENT_WEIGHT = 0.5    # a comment moves a numeric target this far toward what Gemini suggests
NEUTRAL = 5.5           # a rating above this counts as "for" a song's labels, below as "against"
LABEL_MIN_SCORE = 3.0   # how much liking a label needs before it is learned (one 9 = 3.5, one 8 = 2.5)
MAX_LEARNED_TAGS = 4    # most occasions / tags the profile keeps
STEP = 0.5              # how far a target moves toward what the user rated best (0 = never, 1 = all the way)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ratings(ratings, songs=None, shown_ids=None):
    """
    Check the ratings and return a clean {song_id: int rating} dict, or raise ValueError.

    ratings:   {song_id: rating}. Each rating must be a whole number from 1 to 10.
    songs:     optional, the catalog. If given, every id must be a real song.
    shown_ids: optional, the ids that were shown. If given, every one must be rated
               (the Regenerate button only unlocks after all songs are rated).
    """
    if not isinstance(ratings, dict) or not ratings:
        raise ValueError("No ratings were given. Rate the songs from 1 to 10 first.")

    known_ids = {song["id"] for song in songs} if songs is not None else None
    clean = {}
    for song_id, value in ratings.items():
        if known_ids is not None and song_id not in known_ids:
            raise ValueError(f"Song id {song_id!r} is not in the catalog.")
        # bool is a subclass of int in Python, so True would otherwise pass as a rating of 1
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise ValueError(f"Rating for song {song_id} must be a whole number, got {value!r}.")
        value = int(value)
        if not MIN_RATING <= value <= MAX_RATING:
            raise ValueError(f"Rating for song {song_id} must be {MIN_RATING}-{MAX_RATING}, got {value}.")
        clean[song_id] = value

    if shown_ids is not None:
        unrated = [song_id for song_id in shown_ids if song_id not in clean]
        if unrated:
            raise ValueError(f"Rate every song first. Still unrated: {unrated}.")
    return clean


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def compute_metrics(ratings):
    """
    Numbers that say how well the recommendations did. `ratings` must already be validated.

    mean       average rating, 1-10
    hit_rate   share of songs rated 7 or more (0-1), the "precision at k" of this round
    liked      ids rated 7 or more
    disliked   ids rated 4 or less
    """
    values = list(ratings.values())
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 2),
        "hit_rate": round(sum(1 for v in values if v >= LIKE_AT_LEAST) / len(values), 2),
        "liked": [i for i, v in ratings.items() if v >= LIKE_AT_LEAST],
        "disliked": [i for i, v in ratings.items() if v <= DISLIKE_AT_MOST],
    }


def headline(metrics):
    """One sentence for the user, based on the average rating."""
    mean = metrics["mean"]
    if mean >= 8:
        return "The AI did an excellent job recommending songs you love!"
    if mean >= 5:
        return "The AI did a good job recommending songs you like."
    return "The AI needs to improve its recommendations."


# ---------------------------------------------------------------------------
# Improving the profile
# ---------------------------------------------------------------------------

def _limits(feature):
    """(lowest, highest) allowed value for a numeric target. None means no upper limit."""
    if feature == "popularity":
        return 0.0, 100.0
    if feature in ("tempo", "duration"):
        return 1.0, None
    return 0.0, 1.0


def update_profile(profile, ratings, songs):
    """
    Return a NEW profile whose numeric targets (target_energy, target_valence, ...) have
    moved toward the songs rated better than this round's average and away from the
    songs rated worse.

        weight of a song = its rating - the round's average rating
        target = target + STEP * (sum of weight * (song value - target)) / (sum of |weight|)

    Comparing with the round's own average means there is always a direction when the
    ratings differ, even if every song was rated low ("less bad" still points somewhere).
    If every rating is equal there is nothing to compare, so the profile is unchanged.

    Only targets already in the profile are changed. Nothing is invented, the input
    profile is not modified, and labels (mood, genre, occasions) are left alone.
    """
    songs_by_id = {song["id"]: song for song in songs}
    average = sum(ratings.values()) / len(ratings)
    weights = {song_id: rating - average for song_id, rating in ratings.items() if song_id in songs_by_id}
    new_profile = dict(profile)

    for key, target in profile.items():
        if not key.startswith("target_") or isinstance(target, bool) or not isinstance(target, (int, float)):
            continue
        feature = key[len("target_"):]
        field = SONG_FIELD_FOR_FEATURE.get(feature, feature)

        pulls = [(weight, float(songs_by_id[song_id][field]))
                 for song_id, weight in weights.items() if field in songs_by_id[song_id]]
        total = sum(abs(weight) for weight, _value in pulls)
        if total == 0:
            continue
        shift = sum(weight * (value - target) for weight, value in pulls) / total

        low, high = _limits(feature)
        value = max(low, float(target) + STEP * shift)
        if high is not None:
            value = min(high, value)
        new_profile[key] = round(value, 3)
    return new_profile


# ---------------------------------------------------------------------------
# Learning labels from the whole rating history
# ---------------------------------------------------------------------------

# profile key -> song column (single label) and profile key -> song column (several labels, "a|b")
LABEL_COLUMNS = {"favorite_genre": "genre", "favorite_mood": "mood", "language": "language"}
LIST_COLUMNS = {"preferred_occasions": "occasion_tags", "preferred_tags": "mood_tags"}


def _label_scores(history, songs_by_id, column, split):
    """{label: sum of (rating - NEUTRAL) over the rated songs that have it}."""
    scores = {}
    for song_id, rating in history.items():
        song = songs_by_id.get(song_id)
        if not song:
            continue
        raw = str(song.get(column, ""))
        for label in (raw.split("|") if split else [raw]):
            label = label.strip().lower()
            if label and label != "none":
                scores[label] = scores.get(label, 0.0) + (rating - NEUTRAL)
    return scores


def learn_labels(profile, history, songs):
    """
    Learn mood, genre, language, occasions and tags from EVERY rating so far (the history),
    because the numeric update in update_profile cannot change labels.

        single label (mood, genre, language):
            unknown      -> use the best-liked label, if liked enough
            already set  -> replace it only when the user's ratings went against it
                            (its total is negative) and another label is liked enough
        lists (occasions, tags): add the best-liked labels (up to MAX_LEARNED_TAGS)

    A label's total is the sum of (rating - 5.5) over the rated songs that have it, so
    many songs rated high add up and one low rating can cancel a high one.

    Returns (new_profile, notes). The input is not modified.
    """
    songs_by_id = {song["id"]: song for song in songs}
    new_profile = dict(profile)
    notes = []

    for key, column in LABEL_COLUMNS.items():
        if key == "language" and new_profile.get("only_languages"):
            continue                       # the user already fixed the languages
        scores = _label_scores(history, songs_by_id, column, split=False)
        if not scores:
            continue
        best = max(scores, key=scores.get)
        if scores[best] < LABEL_MIN_SCORE:
            continue
        current = new_profile.get(key)
        if current is None:
            new_profile[key] = best
            notes.append(f"Learned from your ratings: {key.replace('_', ' ')} = {best}.")
        elif current != best and scores.get(str(current).lower(), 0.0) < 0:
            new_profile[key] = best
            notes.append(f"Your ratings went against {current}, so {key.replace('_', ' ')} is now {best}.")

    for key, column in LIST_COLUMNS.items():
        scores = _label_scores(history, songs_by_id, column, split=True)
        liked = [label for label, total in sorted(scores.items(), key=lambda item: -item[1])
                 if total >= LABEL_MIN_SCORE]
        current = list(new_profile.get(key) or [])
        added = [label for label in liked if label not in current][:max(0, MAX_LEARNED_TAGS - len(current))]
        if added:
            new_profile[key] = current + added
            notes.append(f"Learned from your ratings: added {key.replace('_', ' ')} {', '.join(added)}.")
    return new_profile, notes


# ---------------------------------------------------------------------------
# Comments (the one place Gemini helps)
# ---------------------------------------------------------------------------

def apply_comments(profile, comments, songs, llm=None, snippets=None, vocab=None, ratings=None):
    """
    Turn comments ({song_id: "too slow"}) into changes to the profile.

    llm: anything with extract_preferences(). If omitted, a GeminiClient is created here
         (not as a default argument, which would be built at import time and crash
         without an API key).

    ratings (optional) tells Gemini whether each commented song was liked.
    Each comment goes through llm.extract_preferences, the same function the interview
    uses, and the result is checked by clean_preferences. Numeric targets that already
    exist move only part of the way (COMMENT_WEIGHT) toward the suggestion; labels and
    new preferences are set directly. A comment that is blank, blocked
    by the input guardrail, or fails (Gemini down) is skipped and never breaks the round.

    Returns (new_profile, notes) where notes is a list of readable lines about what happened.
    """
    songs_by_id = {song["id"]: song for song in songs}
    new_profile = dict(profile)
    notes = []
    if not any((c or "").strip() for c in (comments or {}).values()):
        return new_profile, notes           # nothing to read, so no Gemini needed
    if llm is None:
        try:
            llm = GeminiClient()
        except RuntimeError:                # no API key: the ratings still work, comments are skipped
            return new_profile, ["Comments were ignored (no GEMINI_API_KEY)."]
    for song_id, comment in (comments or {}).items():
        comment = (comment or "").strip()
        if not comment or song_id not in songs_by_id:
            continue
        if not check_input(comment).allowed:
            notes.append(f"Ignored a comment on song {song_id} (it was blocked by the input check).")
            continue
        song = songs_by_id[song_id]
        # Tell Gemini which song the comment is about and whether it was liked, so "too slow"
        # is read as a complaint to fix, not as the sound the listener wants.
        rating = (ratings or {}).get(song_id)
        liked = rating is not None and rating >= LIKE_AT_LEAST
        text = (f'The listener heard "{song.get("title")}" ({song.get("tempo_bpm")} BPM, '
                f'energy {song.get("energy")}, mood {song.get("mood")}) and rated it {rating}/10. '
                f'Their comment: "{comment}". ')
        text += ("Give the preferences that describe what they liked about it." if liked else
                 "Give the preferences they want INSTEAD: the opposite of what they complained about.")
        try:
            found = llm.extract_preferences(text, snippets or [], vocab)
        except LLMError:
            notes.append(f'Could not read the comment "{comment}" (Gemini unavailable).')
            continue
        found = clean_preferences(found, vocab)
        if found:
            for key, value in found.items():
                if value is None:            # nothing was suggested for this key
                    continue
                old = new_profile.get(key)   # None when the profile has no value yet
                if (key.startswith("target_")
                        and isinstance(old, (int, float)) and not isinstance(old, bool)
                        and isinstance(value, (int, float)) and not isinstance(value, bool)):
                    # Gemini's number can swing from run to run, so only move part of the way.
                    value = round(old + COMMENT_WEIGHT * (value - old), 3)
                new_profile[key] = value
            notes.append(f'Comment "{comment}" changed: {", ".join(sorted(found))}.')
        else:
            notes.append(f'Comment "{comment}" did not change anything.')
    return new_profile, notes


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def regenerate(profile, ratings, songs, shown_ids=None, k=5, exclude_ids=(),
               comments=None, llm=None, snippets=None, vocab=None, history=None):
    """
    Validate the ratings, measure the round, improve the profile, and recommend again.

    exclude_ids: songs from earlier rounds. Together with the songs just rated they are left
                 out, so the new list is new music.
    comments:    optional {song_id: text}. Used only when llm is given (see apply_comments).
    history:     optional {song_id: rating} from EARLIER rounds. Labels are learned from the
                 history plus this round (see learn_labels).
    Raises ValueError if the ratings are not valid (see validate_ratings).

    Returns {"metrics", "headline", "profile", "notes", "recommendations"}.
    """
    ratings = validate_ratings(ratings, songs=songs, shown_ids=shown_ids)
    metrics = compute_metrics(ratings)
    new_profile = update_profile(profile, ratings, songs)       # numbers: code
    notes = []
    if comments and llm is not None:                            # words: Gemini, then checked
        new_profile, notes = apply_comments(new_profile, comments, songs, llm, snippets, vocab, ratings)
    new_profile, learned = learn_labels(new_profile, {**(history or {}), **ratings}, songs)
    notes += learned                                            # labels: code, from all ratings
    seen = set(ratings) | set(exclude_ids)
    unseen = [song for song in songs if song["id"] not in seen]
    return {
        "metrics": metrics,
        "headline": headline(metrics),
        "profile": new_profile,
        "notes": notes,
        "recommendations": recommend_songs(new_profile, unseen, k=k),
    }
