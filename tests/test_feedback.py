import pytest

from src.feedback import (
    compute_metrics, headline, regenerate, update_profile, validate_ratings,
)

SONGS = [
    {"id": 1, "title": "Quiet", "genre": "classical", "mood": "calm", "energy": 0.10, "tempo_bpm": 60, "valence": 0.3},
    {"id": 2, "title": "Soft", "genre": "classical", "mood": "calm", "energy": 0.30, "tempo_bpm": 80, "valence": 0.5},
    {"id": 3, "title": "Loud", "genre": "metal", "mood": "intense", "energy": 0.95, "tempo_bpm": 200, "valence": 0.4},
    {"id": 4, "title": "Fresh", "genre": "classical", "mood": "calm", "energy": 0.20, "tempo_bpm": 70, "valence": 0.4},
]


# ---------- validation ----------

@pytest.mark.parametrize("bad", [0, 11, -3, 5.5, "ten", None, True, [9]])
def test_bad_ratings_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_ratings({1: bad})


def test_empty_ratings_are_rejected():
    with pytest.raises(ValueError, match="No ratings"):
        validate_ratings({})
    with pytest.raises(ValueError):
        validate_ratings(None)


def test_unknown_song_id_is_rejected():
    with pytest.raises(ValueError, match="not in the catalog"):
        validate_ratings({99: 5}, songs=SONGS)


def test_every_shown_song_must_be_rated():
    with pytest.raises(ValueError, match="Still unrated"):
        validate_ratings({1: 9}, songs=SONGS, shown_ids=[1, 2])


def test_valid_ratings_pass_and_whole_floats_become_ints():
    assert validate_ratings({1: 1, 2: 10.0}, songs=SONGS, shown_ids=[1, 2]) == {1: 1, 2: 10}


# ---------- metrics and headline ----------

def test_metrics():
    m = compute_metrics({1: 9, 2: 7, 3: 2, 4: 6})
    assert m["mean"] == 6.0
    assert m["hit_rate"] == 0.5            # two of four rated 7+
    assert m["liked"] == [1, 2] and m["disliked"] == [3]


@pytest.mark.parametrize("ratings, expected", [
    ({1: 10, 2: 9}, "excellent"),
    ({1: 8, 2: 8}, "excellent"),           # exactly 8 counts as excellent
    ({1: 5, 2: 5}, "good"),                # exactly 5 counts as good
    ({1: 4, 2: 4}, "improve"),
    ({1: 1, 2: 10, 3: 1, 4: 1}, "improve"),  # one loved song does not rescue a 3.25 average
])
def test_headline(ratings, expected):
    assert expected in headline(compute_metrics(ratings))


# ---------- profile update ----------
# Rule: weight = rating - round average; target += 0.5 * sum(weight * (song - target)) / sum(|weight|)

def test_target_moves_toward_the_better_rated_song():
    # Song 1 (energy 0.10) rated 9, song 3 (energy 0.95) rated 1. Average 5, weights +4 and -4.
    # shift = (4 * (0.10 - 0.5) + -4 * (0.95 - 0.5)) / 8 = -0.425, so target = 0.5 + 0.5 * -0.425
    new = update_profile({"target_energy": 0.5}, {1: 9, 3: 1}, SONGS)
    assert new["target_energy"] == pytest.approx(0.2875, abs=1e-3)


def test_target_moves_up_when_the_loud_song_is_rated_higher():
    new = update_profile({"target_energy": 0.5}, {1: 2, 3: 9}, SONGS)
    assert new["target_energy"] > 0.5


def test_all_low_ratings_still_point_somewhere():
    # Everything was rated low, but the quiet song less low. The target still moves toward it.
    new = update_profile({"target_energy": 0.5}, {1: 3, 3: 1}, SONGS)
    assert new["target_energy"] < 0.5


def test_equal_ratings_change_nothing():
    assert update_profile({"target_energy": 0.5}, {1: 6, 3: 6}, SONGS) == {"target_energy": 0.5}


def test_a_single_rating_changes_nothing():
    # One song has nothing to be compared with.
    assert update_profile({"target_energy": 0.5}, {1: 9}, SONGS) == {"target_energy": 0.5}


def test_tempo_uses_the_tempo_bpm_column():
    new = update_profile({"target_tempo": 100}, {1: 10, 3: 1}, SONGS)   # liked song is 60 BPM, disliked 200
    assert new["target_tempo"] < 100


def test_nothing_is_invented_and_input_is_not_modified():
    profile = {"favorite_mood": "calm", "target_energy": 0.5}
    before = dict(profile)
    new = update_profile(profile, {1: 9, 3: 1}, SONGS)
    assert set(new) == set(profile)                   # no new keys
    assert new["favorite_mood"] == "calm"             # labels untouched
    assert profile == before                          # original not changed


def test_values_stay_in_range():
    new = update_profile({"target_energy": 0.99, "target_popularity": 99}, {1: 1, 3: 10}, SONGS)
    assert 0 <= new["target_energy"] <= 1
    assert 0 <= new["target_popularity"] <= 100


# ---------- regenerate ----------

def test_regenerate_returns_only_unseen_songs_and_a_new_profile():
    profile = {"favorite_mood": "calm", "target_energy": 0.5}
    result = regenerate(profile, {1: 9, 2: 8, 3: 1}, SONGS, shown_ids=[1, 2, 3], k=5)
    assert [song["id"] for song, _, _ in result["recommendations"]] == [4]   # only song 4 is unseen
    assert result["profile"]["target_energy"] != profile["target_energy"]
    assert result["metrics"]["mean"] == 6.0


def test_regenerate_refuses_incomplete_or_invalid_ratings():
    with pytest.raises(ValueError):
        regenerate({"target_energy": 0.5}, {1: 9}, SONGS, shown_ids=[1, 2])
    with pytest.raises(ValueError):
        regenerate({"target_energy": 0.5}, {1: 0}, SONGS)


# ---------- comments (Gemini proposes, code validates) ----------

from src.feedback import apply_comments
from src.llm import LLMError


class FakeLLM:
    """Returns what extract_preferences would return, without calling Gemini."""

    def __init__(self, result=None, fail=False):
        self.result, self.fail, self.texts = result or {}, fail, []

    def extract_preferences(self, text, snippets, vocab=None):
        if self.fail:
            raise LLMError("down")
        self.texts.append(text)
        return self.result


def test_comment_changes_the_profile_through_validation():
    llm = FakeLLM({"target_tempo": 80, "not_a_real_key": "x"})
    profile, notes = apply_comments({"target_tempo": 140}, {3: "too fast"}, SONGS, llm)
    assert profile["target_tempo"] == 110              # halfway from 140 to Gemini's 80
    assert "not_a_real_key" not in profile            # clean_preferences dropped it
    assert "Loud" in llm.texts[0] and "too fast" in llm.texts[0]   # Gemini is told which song
    assert "target_tempo" in notes[0]


def test_comment_is_skipped_when_gemini_is_down():
    profile, notes = apply_comments({"target_tempo": 140}, {3: "too fast"}, SONGS, FakeLLM(fail=True))
    assert profile == {"target_tempo": 140}
    assert "unavailable" in notes[0]


def test_blank_unknown_and_blocked_comments_never_reach_gemini():
    llm = FakeLLM({"target_tempo": 80})
    comments = {1: "   ", 99: "too slow", 2: "my email is a@b.com"}
    profile, notes = apply_comments({"target_tempo": 140}, comments, SONGS, llm)
    assert llm.texts == [] and profile == {"target_tempo": 140}
    assert len(notes) == 1 and "blocked" in notes[0]


def test_regenerate_applies_comments_after_the_arithmetic():
    llm = FakeLLM({"target_tempo": 70})
    result = regenerate({"target_tempo": 140}, {1: 9, 2: 8, 3: 1}, SONGS,
                        comments={3: "too fast"}, llm=llm)
    # The ratings move 140 to about 107 first; the comment then moves it halfway toward 70.
    assert result["profile"]["target_tempo"] < 100
    assert result["notes"]


def test_regenerate_ignores_comments_without_an_llm():
    result = regenerate({"target_energy": 0.5}, {1: 9, 2: 8, 3: 1}, SONGS, comments={3: "too loud"})
    assert not any("Comment" in note for note in result["notes"])


def test_regenerate_leaves_out_songs_from_earlier_rounds():
    result = regenerate({"favorite_mood": "calm"}, {3: 5}, SONGS, exclude_ids=[1, 2])
    assert [song["id"] for song, _, _ in result["recommendations"]] == [4]


def test_comment_sets_labels_and_new_preferences_directly():
    llm = FakeLLM({"favorite_mood": "calm", "target_tempo": 80})
    profile, _ = apply_comments({"favorite_mood": "happy"}, {3: "too aggressive"}, SONGS, llm)
    assert profile["favorite_mood"] == "calm"          # labels are replaced
    assert profile["target_tempo"] == 80               # nothing to blend with, so set directly


def test_comment_prompt_says_whether_the_song_was_liked():
    llm = FakeLLM({})
    apply_comments({}, {3: "too fast"}, SONGS, llm, ratings={3: 2})
    apply_comments({}, {1: "love the piano"}, SONGS, llm, ratings={1: 9})
    assert "rated it 2/10" in llm.texts[0] and "INSTEAD" in llm.texts[0]
    assert "rated it 9/10" in llm.texts[1] and "what they liked" in llm.texts[1]


# ---------- learning labels from the whole history ----------

from src.feedback import learn_labels

LABELED = [
    {"id": 1, "title": "A", "genre": "pop", "mood": "happy", "language": "Japanese", "occasion_tags": "morning|party", "mood_tags": "upbeat"},
    {"id": 2, "title": "B", "genre": "pop", "mood": "happy", "language": "English", "occasion_tags": "party", "mood_tags": "fun"},
    {"id": 3, "title": "C", "genre": "metal", "mood": "intense", "language": "English", "occasion_tags": "workout", "mood_tags": "loud"},
]


def test_unknown_labels_are_learned_from_liked_songs():
    profile, notes = learn_labels({}, {1: 9, 2: 8, 3: 2}, LABELED)
    assert profile["favorite_mood"] == "happy" and profile["favorite_genre"] == "pop"
    assert "party" in profile["preferred_occasions"]
    assert "metal" not in str(profile) and "workout" not in str(profile)     # disliked labels are not learned
    assert any("favorite mood" in note for note in notes)


def test_a_single_lukewarm_rating_is_not_enough():
    profile, notes = learn_labels({}, {1: 7}, LABELED)                         # 7 - 5.5 = 1.5 < 3
    assert profile == {} and notes == []


def test_history_accumulates_across_rounds():
    # One rating alone is too weak (2.5), two together pass the threshold.
    assert learn_labels({}, {1: 8}, LABELED)[0] == {}
    assert learn_labels({}, {1: 8, 2: 8}, LABELED)[0]["favorite_mood"] == "happy"


def test_a_stated_label_is_kept_unless_the_ratings_went_against_it():
    kept, _ = learn_labels({"favorite_mood": "intense"}, {1: 9, 2: 9}, LABELED)
    assert kept["favorite_mood"] == "intense"                                  # no song with it was rated low
    changed, notes = learn_labels({"favorite_mood": "intense"}, {1: 9, 2: 9, 3: 1}, LABELED)
    assert changed["favorite_mood"] == "happy" and any("went against" in note for note in notes)


def test_learning_does_not_modify_the_input_or_overfill_lists():
    profile = {"preferred_occasions": ["a", "b", "c"]}
    new, _ = learn_labels(profile, {1: 10, 2: 10}, LABELED)
    assert profile == {"preferred_occasions": ["a", "b", "c"]} and len(new["preferred_occasions"]) <= 4


def test_regenerate_uses_the_earlier_rounds_too():
    result = regenerate({}, {2: 8}, LABELED, history={1: 8})
    assert result["profile"]["favorite_mood"] == "happy"
