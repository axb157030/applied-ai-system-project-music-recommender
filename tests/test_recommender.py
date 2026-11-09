import pytest
from src.recommender import (Recommender, Song, UserProfile,
                             load_songs, recommend_songs, score_song)

# Shared profile used by the ranking/selection tests: a calm, quiet study playlist.
PROFILE = {
        "genre": "classical", "mood": "calm",
        "target_energy": 0.2, "target_acousticness": 0.9,
        "preferred_occasions": ["study"],
    }

# Happy path: five results, each a (dict, float score, str explanation) tuple.
def test_recommend_songs():
    songs = load_songs("data/songs.csv")
    profile = {
        "genre": "classical", "mood": "calm",
        "target_energy": 0.2, "target_acousticness": 0.9,
        "preferred_occasions": ["study"],
    }

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 5
    for song, score, reasons in recommendations:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, str)

# Class-based path: takes Song/UserProfile objects, returns Song objects, and can explain a pick.
def test_recommender_class():
    # load_songs gives dicts; the class wants Song objects
    songs = [Song(**row) for row in load_songs("data/songs.csv")]
    user = UserProfile(
        favorite_genre="classical",
        favorite_mood="calm",
        target_energy=0.2,
        target_acousticness=0.9,
        preferred_occasions=["study"],
    )

    recommender = Recommender(songs)
    recommendations = recommender.recommend(user, k=5)

    assert len(recommendations) == 5
    for song in recommendations:
        assert isinstance(song, Song)
        explanation = recommender.explain_recommendation(user, song)
        assert isinstance(explanation, str) and explanation

# No preferences: nothing is scored (every score is 0.0) but k songs still come back.
def test_recommend_songs_with_empty_profile():
    songs = load_songs("data/songs.csv")
    profile = {}

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 5
    for song, score, reasons in recommendations:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, str)

# No catalog: an empty result, not an error.
def test_recommend_songs_with_no_songs():
    songs = []
    profile = {
        "genre": "classical", "mood": "calm",
        "target_energy": 0.2, "target_acousticness": 0.9,
        "preferred_occasions": ["study"],
    }

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 0

# Asking for more songs than exist returns the whole catalog.
def test_recommend_songs_with_k_greater_than_song_count():
    songs = load_songs("data/songs.csv")
    profile = {
        "genre": "classical", "mood": "calm",
        "target_energy": 0.2, "target_acousticness": 0.9,
        "preferred_occasions": ["study"],
    }

    recommendations = recommend_songs(profile, songs, k=100)

    assert len(recommendations) == len(songs)
    for song, score, reasons in recommendations:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, str)

# Work later on this and fine-tune the scoring and recommendation logic to handle invalid profiles gracefully.
# Unknown labels and out-of-range numbers currently just score low (nothing is validated).
# If range validation is added, the -1.0 / 2.0 values would raise ValueError instead.
def test_recommend_songs_with_invalid_profile():
    songs = load_songs("data/songs.csv")
    profile = {
        "genre": "unknown_genre", "mood": "unknown_mood",
        "target_energy": -1.0, "target_acousticness": 2.0,
        "preferred_occasions": ["unknown_occasion"],
    }

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 5
    for song, score, reasons in recommendations:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, str)

# Partial profile: only requested features are scored.
# Explanation = one entry per feature + the "total:" line, so semicolons == number of
# features. That equals len(profile) only because every key here is a scored feature.
def test_recommend_songs_with_partial_profile():
    songs = load_songs("data/songs.csv")
    profile = {
        "genre": "classical",
        "target_energy": 0.2,
    }

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 5
    for song, score, reasons in recommendations:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, str)
        assert reasons.count(";") == len(profile)  # Each profile attribute should contribute to the explanation

# Same as the no-songs test above (duplicate; one of the two can be removed).
def test_recommend_songs_with_empty_song_list():
    songs = []
    profile = {
        "genre": "classical", "mood": "calm",
        "target_energy": 0.2, "target_acousticness": 0.9,
        "preferred_occasions": ["study"],
    }

    recommendations = recommend_songs(profile, songs, k=5)

    assert len(recommendations) == 0

# A None profile is not supported: is_allowed() calls user_prefs.get() and raises AttributeError.
def test_recommend_songs_rejects_none_profile():
    with pytest.raises(AttributeError):
        recommend_songs(None, load_songs("data/songs.csv"), k=5) # pyright: ignore[reportArgumentType]

# Selection check: no song outside the top 5 outscores the worst picked song.
# artist_penalty=0 so the picked scores are raw and comparable with score_song().
def test__no_song_out_of_top_5_has_higher_score_then_those_top_5_songs():
    songs = load_songs("data/songs.csv")
    results = recommend_songs(PROFILE, songs, k=5, artist_penalty=0)
    picked = {s["id"] for s, _, _ in results}
    worst_pick = min(score for _, score, _ in results)
    for song in songs:
        if song["id"] not in picked:
            assert score_song(PROFILE, song)[0] <= worst_pick + 1e-9