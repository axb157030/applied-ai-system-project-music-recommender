import csv
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

from .ScoreCard import ScoreCard

# Part of the genre weight earned when two genre labels share text (e.g. "pop" and "indie pop").
SIMILAR_GENRE_CREDIT = 0.75

# How far apart two values can be before their similarity drops to zero.
# Each range MUST match the scale of the attribute it is used for. A 0-100 attribute
# compared with a range of 1.0 would score 0 for almost every song.
POPULARITY_RANGE = 100      # popularity is on a 0-100 scale
DECADE_RANGE = 50           # five decades apart earns no credit
DURATION_RANGE_SEC = 180    # three minutes apart earns no credit
TEMPO_RANGE_BPM = 100       # assumption: 100 BPM apart earns no credit, adjust to taste
DEFAULT_RANGE = 1.0         # energy, acousticness, valence, ... are all on a 0-1 scale

# Default size of the "variety" penalty applied to each repeat of an artist already picked.
DEFAULT_ARTIST_PENALTY = 0.10

# --- Float preferences ---------------------------------------------------------
# Every float preference in the user's prefs is named "target_<feature>", and <feature>
# is also the key used in the strategy weights ("target_energy" -> weight "energy").
# score_song() loops over those preferences, so a new float feature needs no new code.
# Only the exceptions are listed here, everything else uses the defaults:
#   - the song field is not called <feature>  -> SONG_FIELD_FOR_FEATURE
#   - the attribute is not on a 0-1 scale     -> RANGE_FOR_FEATURE
SONG_FIELD_FOR_FEATURE = {"tempo": "tempo_bpm", "duration": "duration_sec"}
RANGE_FOR_FEATURE = {
    "popularity": POPULARITY_RANGE,
    "tempo": TEMPO_RANGE_BPM,
    "duration": DURATION_RANGE_SEC,
}


@dataclass
class Song:
    """
    Catalog record containing a song's identity and recommendation features.

    Attributes:
        id: Song identifier.
        title: Song title.
        artist: Performing artist.
        genre: Genre label, such as "pop" or "lofi".
        mood: Mood label, such as "happy" or "chill".
        energy: Relative intensity, represented as a decimal feature value.
        tempo_bpm: Tempo in beats per minute.
        valence: Measure of how positive or upbeat the song sounds.
        danceability: Measure of how suitable the song is for dancing.
        acousticness: Measure of how acoustic rather than electronic the song sounds.
        popularity: How widely known the song is, from 0 to 100.
        release_decade: Decade the song came out, written as the first year (e.g. 2010).
        mood_tags: Detailed mood tags separated by "|" (e.g. "study|cozy").
        vocals: "vocal" if the song has singing, "instrumental" if not.
        duration_sec: Song length in seconds.
        instrumentalness: Likelihood the song has no vocals, from 0 to 1.
        speechiness: How much of the song is spoken words, from 0 to 1 (not scored yet).
        language: Language of the lyrics.
        explicit: "yes" or "no". Used as a filter, not as a score.
        occasion_tags: Occasion tags separated by "|" (e.g. "study|relax").
        url: Link to the song.
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: float = 50.0
    release_decade: int = 2020
    mood_tags: str = ""
    vocals: str = "vocal"
    duration_sec: float = 200.0
    instrumentalness: float = 0.0
    speechiness: float = 0.0
    language: str = "English"
    explicit: str = "no"          # CSV has "yes"/"no"
    occasion_tags: str = ""       # "study|relax", same format as mood_tags
    url: str = ""


@dataclass
class UserProfile:
    """
    Preferences used to compare a user with songs in the catalog.

    Attributes:
        favorite_genre: Preferred genre label.
        favorite_mood: Preferred mood label.
        target_energy: Desired energy feature value.
        target_acousticness: Desired acousticness feature value.
        target_popularity: Optional desired popularity (0-100).
        preferred_decade: Optional preferred release decade (e.g. 1990).
        preferred_tags: Optional list of wanted mood tags (e.g. ["study", "cozy"]).
        vocals: Optional "vocal" or "instrumental" preference.
        target_duration: Optional desired song length in seconds.
        preferred_occasions: Optional list of wanted occasions (e.g. ["study", "workout"]).
        language: Optional preferred lyrics language.
        allow_explicit: False removes explicit songs; None or True keeps them.
        target_valence: Optional desired positivity, sad (0) to happy (1).
        target_tempo: Optional desired tempo in BPM.
        target_danceability: Optional desired danceability (0-1).
        target_instrumentalness: Optional desired instrumentalness (0-1).
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_acousticness: float
    target_popularity: Optional[float] = None
    preferred_decade: Optional[int] = None
    preferred_tags: Optional[List[str]] = None
    vocals: Optional[str] = None
    target_duration: Optional[float] = None
    preferred_occasions: Optional[List[str]] = None   # ["study", "workout"]
    language: Optional[str] = None
    allow_explicit: Optional[bool] = None
    target_valence: Optional[float] = None            # sad (0) to happy (1)
    target_tempo: Optional[float] = None              # BPM
    target_danceability: Optional[float] = None
    target_instrumentalness: Optional[float] = None

    def preferences(self) -> Dict:
        """Return the profile as the preference dict that score_song understands."""
        prefs = {
            "genre": self.favorite_genre,
            "mood": self.favorite_mood,
            # score_song only reads numeric keys that start with "target_".
            "target_energy": self.target_energy,
            "target_acousticness": self.target_acousticness,
            "target_popularity": self.target_popularity,
            "preferred_decade": self.preferred_decade,
            "preferred_tags": self.preferred_tags,
            "vocals": self.vocals,
            "target_duration": self.target_duration,
            "preferred_occasions": self.preferred_occasions,
            "language": self.language,
            "allow_explicit": self.allow_explicit,
            "target_valence": self.target_valence,
            "target_tempo": self.target_tempo,
            "target_danceability": self.target_danceability,
            "target_instrumentalness": self.target_instrumentalness,
        }
        # Drop unset preferences so score_song only scores what the user actually asked for.
        # False is kept on purpose: allow_explicit=False is a real choice, not "no preference".
        return {key: value for key, value in prefs.items() if value is not None}


# --- Ranking strategies --------------------------------------------------------
#
# Strategy pattern: every ranking mode is a RankingStrategy holding its own weights.
# score_song() is the one shared scoring routine; it takes a weights dict, so switching
# modes never changes scoring code. To add a mode, add one RankingStrategy to STRATEGIES.
#
# A feature is only scored if its key exists in the weights. The keys must match what
# score_song passes to card.add(): "genre", "mood", "language", "decade", "tags",
# "vocals", "preferred_occasions", and the <feature> part of every "target_<feature>".
# The "explicit" weight is never used: explicit songs are removed by is_allowed(), not scored.

@dataclass(frozen=True)
class RankingStrategy:
    """A named way of ranking songs: how much each feature matters."""
    name: str
    description: str
    weights: Dict[str, float]


STRATEGIES: Dict[str, RankingStrategy] = {
    strategy.name: strategy
    for strategy in (
        RankingStrategy(
            "balanced",
            "Genre and mood matter most, energy and acousticness add detail.",
            {"genre": 0.35, "mood": 0.30, "energy": 0.20, "acousticness": 0.15,
             "popularity": 0.10, "decade": 0.10, "tags": 0.15, "vocals": 0.10, "duration": 0.05,
             "preferred_occasions": 0.10, "language": 0.10, "explicit": 0.10,
             "valence": 0.10, "tempo": 0.10, "danceability": 0.10, "instrumentalness": 0.10},
        ),
        RankingStrategy(
            "genre_first",
            "The song's genre decides almost everything.",
            {"genre": 0.60, "mood": 0.15, "energy": 0.10, "acousticness": 0.05,
             "popularity": 0.05, "decade": 0.05, "tags": 0.05, "vocals": 0.05, "duration": 0.02,
             "preferred_occasions": 0.05, "language": 0.05, "explicit": 0.05,
             "valence": 0.05, "tempo": 0.05, "danceability": 0.05, "instrumentalness": 0.05},
        ),
        RankingStrategy(
            "mood_first",
            "The song's mood and mood tags decide most of the ranking.",
            {"mood": 0.55, "tags": 0.25, "genre": 0.15, "energy": 0.15, "acousticness": 0.10,
             "popularity": 0.05, "decade": 0.05, "vocals": 0.05, "duration": 0.02,
             "preferred_occasions": 0.05, "language": 0.05, "explicit": 0.05,
             "valence": 0.05, "tempo": 0.05, "danceability": 0.05, "instrumentalness": 0.05},
        ),
        RankingStrategy(
            "energy_focused",
            "How close the song's energy is to your target decides most of the ranking.",
            {"energy": 0.55, "mood": 0.15, "genre": 0.10, "acousticness": 0.15,
             "popularity": 0.05, "decade": 0.05, "tags": 0.05, "vocals": 0.05, "duration": 0.02,
             "preferred_occasions": 0.05, "language": 0.05, "explicit": 0.05,
             "valence": 0.05, "tempo": 0.05, "danceability": 0.05, "instrumentalness": 0.05},
        ),
    )
}


def get_strategy(mode: Union[str, RankingStrategy]) -> RankingStrategy:
    """Look up a ranking strategy by name (a RankingStrategy is returned as is)."""
    if isinstance(mode, RankingStrategy):
        return mode
    try:
        return STRATEGIES[mode]
    except KeyError:
        raise ValueError(f"Unknown ranking mode '{mode}'. Choose from: {', '.join(STRATEGIES)}") from None


def is_allowed(song: Dict, user_prefs: Dict) -> bool:
    """Hard filter: False only when the user turned explicit songs off and the song is explicit.

    `is not False` is deliberate. None (no preference) and True both allow every song,
    and only an explicit False filters. Both recommend_songs() and Recommender.recommend()
    call this, so the rule lives in one place.
    """
    if user_prefs.get("allow_explicit") is not False:
        return True
    return str(song.get("explicit", "no")).casefold() != "yes"


class Recommender:
    """
    Content-based recommender that scores songs against one user's preferences.

    Attributes:
        songs: Candidate songs available for recommendation.
        mode: Name of the ranking strategy used to weigh the features.

    Genre and mood matches receive the largest weights in the default mode; the other
    features contribute similarity scores. No other users' listening histories are used.
    """
    def __init__(self, songs: List[Song], mode: str = "balanced"):
        """Store the catalog of candidate songs and the ranking mode."""
        self.songs = songs
        self.mode = mode

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return up to k songs ranked by weighted profile-feature similarity."""
        preferences = user.preferences()
        weights = get_strategy(self.mode).weights
        # This path does not go through recommend_songs(), so it needs its own explicit
        # filter. Songs here are dataclasses, so convert each to a dict for the check.
        candidates = [song for song in self.songs if is_allowed(asdict(song), preferences)]
        ranked_songs = sorted(
            candidates,
            key=lambda song: score_song(preferences, asdict(song), weights)[0],
            reverse=True,
        )
        return ranked_songs[:max(0, k)]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain how the song's attributes compare with the user's preferences."""
        weights = get_strategy(self.mode).weights
        _, reasons = score_song(user.preferences(), asdict(song), weights)
        return "; ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Load song rows from CSV, converting IDs and audio features to numbers."""
    # These columns are read as text, so they are converted to numbers here.
    numeric_fields = ("energy", "tempo_bpm", "valence", "danceability", "acousticness",
                      "popularity", "duration_sec", "instrumentalness", "speechiness")
    with open(csv_path, newline="", encoding="utf-8") as song_file:
        songs = []
        for row in csv.DictReader(song_file):
            row["id"] = int(row["id"])
            for field in numeric_fields:
                # Truthiness check skips both missing columns and empty cells ("").
                if row.get(field):
                    row[field] = float(row[field])
            if "release_decade" in row:
                row["release_decade"] = int(row["release_decade"])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict, weights: Optional[Dict[str, float]] = None) -> Tuple[float, List[str]]:
    """Return a 0-1 content-match score and short explanations for one song.

    `weights` says how much each feature matters (see STRATEGIES); it defaults to the
    "balanced" strategy. Only features the user gave a preference for are scored, and
    the total is divided by the weight of those features, so the score always falls
    between 0 and 1.

    Scoring happens in three groups:
      1. Text labels (genre, mood, language, vocals): exact or partial match.
      2. Tag lists (mood tags, occasions): share of wanted tags the song has.
      3. Numbers: similarity that falls linearly from 1.0 (same) to 0.0 (range apart).
    """
    weights = weights if weights is not None else STRATEGIES["balanced"].weights
    card = ScoreCard(weights)

    # Accept "energy" as a shortcut for "target_energy". Work on a copy so the caller's
    # dict is not changed.
    user_prefs = dict(user_prefs)
    if user_prefs.get("target_energy") is None and user_prefs.get("energy") is not None:
        user_prefs["target_energy"] = user_prefs["energy"]

    # --- 1. Text labels --------------------------------------------------------

    # Genre: exact match = full credit, shared text ("pop" vs "indie pop") = partial credit.
    preferred_genre = user_prefs.get("favorite_genre", user_prefs.get("genre"))
    if preferred_genre is not None and song.get("genre") is not None:
        song_genre = song["genre"]
        # Case-insensitive, so "Pop" matches "indie pop".
        wanted, actual = str(preferred_genre).casefold(), str(song_genre).casefold()
        if wanted == actual:
            card.add("genre", "genre", 1.0, f"exact match with {preferred_genre}")
        elif wanted in actual or actual in wanted:
            card.add("genre", "genre", SIMILAR_GENRE_CREDIT, f"{song_genre} is similar to {preferred_genre}")
        else:
            card.add("genre", "genre", 0.0, f"{song_genre} does not match {preferred_genre}")

    # Mood: all or nothing.
    preferred_mood = user_prefs.get("favorite_mood", user_prefs.get("mood"))
    if preferred_mood is not None and song.get("mood") is not None:
        song_mood = song["mood"]
        if str(preferred_mood).casefold() == str(song_mood).casefold():
            card.add("mood", "mood", 1.0, f"exact match with {preferred_mood}")
        else:
            card.add("mood", "mood", 0.0, f"{song_mood} does not match {preferred_mood}")

    # Language: all or nothing, same rule as mood.
    preferred_language = user_prefs.get("language")
    if preferred_language is not None and song.get("language") is not None:
        song_language = song["language"]
        if str(preferred_language).casefold() == str(song_language).casefold():
            card.add("language", "language", 1.0, f"exact match with {preferred_language}")
        else:
            card.add("language", "language", 0.0, f"{song_language} does not match {preferred_language}")

    # Vocals ("vocal" / "instrumental"): all or nothing. float(True) is 1.0, float(False) is 0.0.
    preferred_vocals = user_prefs.get("vocals")
    if preferred_vocals is not None and song.get("vocals") is not None:
        vocals_match = str(preferred_vocals).casefold() == str(song["vocals"]).casefold()
        card.add("vocals", "vocals", float(vocals_match),
            f"song is {song['vocals']}, you want {preferred_vocals}")

    # --- 2. Tag lists ----------------------------------------------------------
    # Song tags are one string like "study|cozy". Score = wanted tags found / wanted tags.

    preferred_tags = user_prefs.get("preferred_tags")
    if preferred_tags and song.get("mood_tags") is not None:
        song_tags = {tag.casefold() for tag in str(song["mood_tags"]).split("|") if tag}
        matched = [tag for tag in preferred_tags if str(tag).casefold() in song_tags]
        card.add("tags", "tags", len(matched) / len(preferred_tags),
            f"{len(matched)} of {len(preferred_tags)} wanted tags found ({', '.join(matched) or 'none'})")

    # Occasions work exactly like tags, just against the song's occasion_tags field.
    # The first card.add argument must equal the weight key in STRATEGIES, which is
    # "preferred_occasions" (not "occasions").
    preferred_occasions = user_prefs.get("preferred_occasions")
    if preferred_occasions and song.get("occasion_tags") is not None:
        song_occasions = {t.casefold() for t in str(song["occasion_tags"]).split("|") if t}
        matched = [o for o in preferred_occasions if str(o).casefold() in song_occasions]
        card.add("preferred_occasions", "occasions", len(matched) / len(preferred_occasions),
            f"{len(matched)} of {len(preferred_occasions)} wanted occasions found ({', '.join(matched) or 'none'})")

    # --- 3. Numbers ------------------------------------------------------------

    # Decade is the one whole-number (int) feature, so it keeps its own small block.
    preferred_decade = user_prefs.get("preferred_decade")
    if preferred_decade is not None and song.get("release_decade") is not None:
        decade_similarity = max(0.0, 1.0 - abs(int(preferred_decade) - int(song["release_decade"])) / DECADE_RANGE)
        card.add("decade", "decade", decade_similarity,
            f"song from the {int(song['release_decade'])}s vs wanted {int(preferred_decade)}s, "
            f"similarity {decade_similarity:.2f}")

    # Every float preference ("target_energy", "target_tempo", ...) goes through one loop.
    for pref_key, target_value in user_prefs.items():
        if not pref_key.startswith("target_"):
            continue

        feature = pref_key[len("target_"):]          # "target_tempo" -> "tempo"
        if feature not in weights:
            continue                                  # this strategy does not weigh it

        # Look up the song field ("tempo" lives in "tempo_bpm", "duration" in "duration_sec").
        song_value = song.get(SONG_FIELD_FOR_FEATURE.get(feature, feature))
        if song_value is None:
            continue
        try:
            song_number, target_number = float(song_value), float(target_value)
        except (TypeError, ValueError):
            continue                                  # missing, empty or non-numeric value: skip

        similarity, reason = __get_score_and_reason_for_float_attribute(
            song_number, target_number, RANGE_FOR_FEATURE.get(feature, DEFAULT_RANGE), feature)
        card.add(feature, feature, similarity, reason)

    # Final score = weighted average of the features that were scored.
    # Dividing by the weight of scored features (not all weights) keeps the result between 0 and 1
    # and stops a missing preference from lowering the score.
    total_weight = sum(weight for weight, _ in card.components)
    if not total_weight:
        return 0.0, ["no matching profile features were provided"]
    earned = sum(weight * similarity for weight, similarity in card.components)
    score = earned / total_weight
    card.reasons.append(f"total: {earned:.3f} of {total_weight:.2f} possible = {score:.2f}")
    return score, card.reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: Union[str, RankingStrategy] = "balanced",
    artist_penalty: float = DEFAULT_ARTIST_PENALTY,
) -> List[Tuple[Dict, float, str]]:
    """Return up to k songs as (song, score, explanation) tuples, best first.

    `mode` picks the ranking strategy. `artist_penalty` is subtracted from a song's score
    once for every song by the same artist that was already picked, so one artist cannot
    fill the whole list; use 0 to turn the variety rule off.
    """
    # Filter first: explicit songs are removed, not scored, and this runs once per request.
    songs = [song for song in songs if is_allowed(song, user_prefs)]

    weights = get_strategy(mode).weights
    scored = [
        (song, score, reasons)
        for song in songs
        for score, reasons in [score_song(user_prefs, song, weights)]
    ]
    scored.sort(key=lambda result: result[1], reverse=True)

    # Greedy pick: each round takes the best remaining song after subtracting a penalty
    # for every song by the same artist that was already picked.
    picked = []
    artist_counts = Counter()
    remaining = scored
    while remaining and len(picked) < max(0, k):
        def adjusted(item):
            return item[1] - artist_penalty * artist_counts[item[0].get("artist")]

        best_index = max(range(len(remaining)), key=lambda i: adjusted(remaining[i]))
        song, score, reasons = remaining.pop(best_index)
        artist = song.get("artist")
        repeats = artist_counts[artist]
        final_score = score
        if artist_penalty and repeats:
            penalty = artist_penalty * repeats
            final_score = max(0.0, score - penalty)
            artist_name = str(artist).replace(";", ",")
            reasons = reasons + [f"variety: {repeats} earlier pick(s) by {artist_name} (-{penalty:.3f})"]
        artist_counts[artist] += 1
        picked.append((song, final_score, "; ".join(reasons)))
    return picked


def __calculate_score_of_float_attribute(song_value: float, target_value: float, range_value: float) -> float:
    """Calculate the similarity score of a song's attribute compared to the user's target value.

    The similarity score is calculated as 1 minus the absolute difference between the song's
    attribute value and the user's target value, divided by the specified range. The score is
    capped at a minimum of 0.0.
    """
    return max(0.0, 1.0 - abs(song_value - target_value) / range_value)


def __get_score_and_reason_for_float_attribute(song_value: float, target_value: float, range_value: float, attribute_name: str) -> Tuple[float, str]:
    """Calculate the similarity score and reason for a song's attribute compared to the user's target value.

    The similarity score is calculated as 1 minus the absolute difference between the song's
    attribute value and the user's target value, divided by the specified range. The score is
    capped at a minimum of 0.0. The reason string provides details about the comparison.
    """
    similarity_score = __calculate_score_of_float_attribute(song_value, target_value, range_value)
    reason = f"song {attribute_name} {float(song_value):.2f} vs target {float(target_value):.2f}, similarity {similarity_score:.2f}"
    return similarity_score, reason