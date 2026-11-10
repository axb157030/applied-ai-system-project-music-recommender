from src.preferences import clean_preferences, enough_to_recommend, parse_preferences_json, unknown_attributes


def test_valid_values_are_kept():
    raw = {"favorite_genre": "Classical", "target_energy": 0.2, "vocals": "instrumental",
           "preferred_occasions": ["Study"], "allow_explicit": False}
    assert clean_preferences(raw) == {"favorite_genre": "classical", "target_energy": 0.2,
                                      "vocals": "instrumental", "preferred_occasions": ["study"],
                                      "allow_explicit": False}


def test_out_of_range_number_is_dropped_not_clamped():
    assert clean_preferences({"target_energy": 7.0}) == {}      # "energy 700%" guardrail
    assert clean_preferences({"target_popularity": 150}) == {}


def test_unknown_keys_and_nulls_are_ignored():
    assert clean_preferences({"hack": "x", "target_energy": None}) == {}


def test_wrong_types_are_dropped():
    raw = {"target_energy": "loud", "vocals": "maybe", "allow_explicit": "no", "preferred_tags": "calm"}
    assert clean_preferences(raw) == {}


def test_non_dict_gives_empty():
    assert clean_preferences(["a"]) == {}
    assert clean_preferences(None) == {}


def test_any_two_preferences_are_enough():
    assert enough_to_recommend({"target_energy": 0.8, "preferred_occasions": ["workout"]})
    assert enough_to_recommend({"favorite_genre": "jazz", "vocals": "instrumental"})


def test_one_preference_is_not_enough():
    assert not enough_to_recommend({"favorite_genre": "jazz"})
    assert not enough_to_recommend({})


def test_explicit_filter_alone_does_not_count_as_a_taste():
    assert not enough_to_recommend({"allow_explicit": False, "favorite_genre": "jazz"})


def test_unknown_attributes_skips_what_is_known_and_keeps_priority_order():
    unknown = unknown_attributes({"favorite_mood": "calm"})
    assert "favorite_mood" not in unknown
    assert unknown[0] == "preferred_occasions"


def test_json_in_code_fence_is_parsed():
    assert parse_preferences_json('```json\n{"target_energy": 0.3}\n```') == {"target_energy": 0.3}


def test_garbage_reply_gives_empty():
    assert parse_preferences_json("Sure! Here you go") == {}
    assert parse_preferences_json("") == {}


# ---------- label vocabulary check (catalog words only) ----------

SONGS = [{"genre": "classical", "mood": "calm", "occasion_tags": "study|sleep", "mood_tags": "dreamy|gentle"}]


def test_build_vocab_reads_catalog_labels():
    from src.preferences import build_vocab
    v = build_vocab(SONGS)
    assert v["favorite_genre"] == {"classical"} and v["preferred_occasions"] == {"study", "sleep"}


def test_invented_labels_are_dropped_with_vocab():
    from src.preferences import build_vocab
    raw = {"favorite_genre": "death polka", "favorite_mood": "calm",
           "preferred_occasions": ["study or focus", "sleep"]}
    assert clean_preferences(raw, build_vocab(SONGS)) == {"favorite_mood": "calm", "preferred_occasions": ["sleep"]}


def test_without_vocab_labels_are_kept():
    assert clean_preferences({"favorite_genre": "death polka"}) == {"favorite_genre": "death polka"}


# ---------- songs the user names ----------

from src.preferences import clean_preferences, find_song, seed_preferences, songs_by_artist

CATALOG = [
    {"id": 1, "title": "[Rainych] Hotaru no Hikari - Ikimono Gakari (cover)", "artist": "Rainych Ran",
     "genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "tempo_bpm": 180,
     "occasion_tags": "singalong|morning", "mood_tags": "anime|upbeat", "vocals": "vocal", "language": "Japanese"},
    {"id": 2, "title": "Clair de Lune", "artist": "Claude Debussy", "genre": "classical", "mood": "calm",
     "energy": 0.1, "valence": 0.3, "tempo_bpm": 60, "occasion_tags": "sleep", "mood_tags": "gentle",
     "vocals": "instrumental", "language": "none"},
]


def test_song_lists_are_kept_as_trimmed_strings_and_capped():
    cleaned = clean_preferences({"liked_songs": [" Clair de Lune ", "", 5, None] + ["x"] * 10, "disliked_songs": "no"})
    assert cleaned["liked_songs"][0] == "Clair de Lune"
    assert len(cleaned["liked_songs"]) == 5
    assert "disliked_songs" not in cleaned          # not a list


def test_find_song_matches_titles_despite_typos_and_extra_words():
    assert find_song("clare de lune", CATALOG)["id"] == 2
    assert find_song("Hotaru no Hikari", CATALOG)["id"] == 1           # shorter than the catalog title
    assert find_song("my favourite is Clair de Lune by Debussy", CATALOG)["id"] == 2


def test_find_song_returns_none_for_unknown_or_empty_text():
    assert find_song("Sha La La", CATALOG) is None
    assert find_song("", CATALOG) is None
    assert find_song("the weather in Paris", CATALOG) is None


def test_songs_by_artist_handles_typos_and_ignores_short_words():
    assert [s["id"] for s in songs_by_artist("Sha La La by Rainychy", CATALOG)] == [1]
    assert songs_by_artist("by the sea", CATALOG) == []


def test_seed_preferences_describe_the_named_songs():
    prefs = seed_preferences([CATALOG[0]])
    assert prefs["favorite_mood"] == "happy" and prefs["favorite_genre"] == "pop"
    assert prefs["language"] == "japanese" and prefs["target_tempo"] == 180.0
    assert prefs["preferred_occasions"] == ["singalong", "morning"]


def test_seed_preferences_average_numbers_and_skip_language_none():
    prefs = seed_preferences(CATALOG)
    assert prefs["target_energy"] == 0.45
    assert prefs["language"] == "japanese"                              # "none" never counts
    assert "vocals" not in prefs                                         # the songs disagree
    assert seed_preferences([]) == {}
