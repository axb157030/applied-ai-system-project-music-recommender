import os

from src.interview import Interview, FALLBACK_QUESTIONS, NOT_IN_DOCS, GEMINI_DOWN
from src.llm import LLMError

SONGS = [
    {"id": 1, "title": "Quiet Library", "artist": "Ana", "genre": "lofi", "mood": "chill",
     "energy": 0.2, "occasion_tags": "study|relax", "mood_tags": "calm", "explicit": "no"},
    {"id": 2, "title": "Iron Lungs", "artist": "Bo", "genre": "metal", "mood": "intense",
     "energy": 0.95, "occasion_tags": "workout", "mood_tags": "loud", "explicit": "no"},
]

CHILL = {"favorite_mood": "chill"}
STUDY = {"preferred_occasions": ["study"]}


class FakeRetriever:
    """Stands in for Retriever: no files, no Gemini."""

    def retrieve(self, query, top_k=3):
        return [("attributes.md", f"reference about {query}")]

    def answer_rag(self, query, top_k=3):
        if "energy" in query.lower():
            return "Energy is how intense a song feels. (Sources: attributes.md)"
        return "I do not know based on these docs."


class FakeLLM:
    """Stands in for GeminiClient. `extractions` are returned one per extract_preferences call."""

    def __init__(self, extractions=(), fail_ask=False, fail_extract=False):
        self.extractions = list(extractions)
        self.fail_ask = fail_ask
        self.fail_extract = fail_extract
        self.extract_calls = []

    def ask(self, prompt, as_json=False, temperature=None):
        if self.fail_ask:
            raise LLMError("down")
        return "Fake question?"

    def extract_preferences(self, user_text, snippets, vocab=None):
        if self.fail_extract:
            raise LLMError("down")
        self.extract_calls.append(user_text)
        return self.extractions.pop(0) if self.extractions else {}


def converse(replies, llm, tmp_path, retriever=None):
    """Run a scripted conversation. Returns (interview, everything said, prompts shown)."""
    replies = iter(replies)
    said, prompts = [], []

    def ask_user(prompt):
        prompts.append(prompt)
        return next(replies)

    interview = Interview(SONGS, retriever or FakeRetriever(), llm,
                              ask_user=ask_user, say=said.append, log_dir=str(tmp_path))
    interview.run()
    return interview, "\n".join(said), prompts


def test_two_preferences_in_one_reply_gives_recommendations(tmp_path):
    llm = FakeLLM([{**CHILL, **STUDY}])
    iv, said, _ = converse(["calm songs for studying", "quit"], llm, tmp_path)
    assert iv.last_recommendations[0][0]["title"] == "Quiet Library"
    assert "Quiet Library" in said


def test_side_question_is_answered_then_the_question_is_asked_again(tmp_path):
    llm = FakeLLM([CHILL, STUDY])
    iv, said, prompts = converse(
        ["something chill", "what is energy?", "for studying", "quit"], llm, tmp_path)
    assert "Energy is how intense" in said
    # The side question was NOT treated as a preference...
    assert llm.extract_calls == ["something chill", "for studying"]
    # ...and the same interview question was shown again afterwards.
    assert prompts[1] == prompts[2]
    assert iv.last_recommendations


def test_question_without_question_mark_falls_back_to_docs(tmp_path):
    llm = FakeLLM([CHILL])  # second reply extracts nothing
    iv, said, _ = converse(["something chill", "what is energy", "quit"], llm, tmp_path)
    assert "Energy is how intense" in said
    assert "preferred_occasions" not in iv.skipped  # still pending, not skipped


def test_unusable_reply_is_skipped_instead_of_looping_forever(tmp_path):
    llm = FakeLLM([CHILL])
    iv, _, _ = converse(["something chill", "banana", "quit"], llm, tmp_path)
    assert "preferred_occasions" in iv.skipped
    assert iv.pending_attr == "target_energy"  # moved on to the next attribute


def test_skip_command(tmp_path):
    llm = FakeLLM([CHILL])
    iv, _, _ = converse(["something chill", "no idea", "quit"], llm, tmp_path)
    assert "preferred_occasions" in iv.skipped


def test_too_many_side_questions_moves_on(tmp_path):
    llm = FakeLLM([CHILL])
    iv, _, _ = converse(
        ["something chill", "what is energy?", "what is energy?", "what is energy?", "quit"],
        llm, tmp_path)
    assert "preferred_occasions" in iv.skipped


def test_question_the_docs_cannot_answer_gets_a_friendly_message(tmp_path):
    llm = FakeLLM()
    _, said, _ = converse(["who won the game?", "quit"], llm, tmp_path)
    assert NOT_IN_DOCS in said


def test_fallback_question_when_gemini_cannot_write_one(tmp_path):
    llm = FakeLLM([CHILL], fail_ask=True)
    _, _, prompts = converse(["something chill", "quit"], llm, tmp_path)
    assert FALLBACK_QUESTIONS["preferred_occasions"] in prompts[1]


def test_gemini_failure_while_extracting_does_not_crash(tmp_path):
    llm = FakeLLM(fail_extract=True)
    iv, said, _ = converse(["something chill", "quit"], llm, tmp_path)
    assert GEMINI_DOWN in said
    assert iv.profile == {}


def test_invalid_values_from_the_model_never_reach_the_profile(tmp_path):
    llm = FakeLLM([{"favorite_mood": "nonexistent mood", "target_energy": 5}])
    iv, _, _ = converse(["whatever I guess", "quit"], llm, tmp_path)
    assert iv.profile == {}


def test_recommend_with_no_preferences_asks_for_some(tmp_path):
    llm = FakeLLM()
    iv, said, _ = converse(["recommend", "quit"], llm, tmp_path)
    assert "don't know your taste" in said
    assert iv.last_recommendations == []


def test_reset_clears_the_profile(tmp_path):
    llm = FakeLLM([{**CHILL, **STUDY}])
    iv, _, _ = converse(["calm songs for studying", "reset", "quit"], llm, tmp_path)
    assert iv.profile == {}
    assert iv.last_recommendations == []


def test_trace_is_written_to_the_log_folder(tmp_path):
    llm = FakeLLM([{**CHILL, **STUDY}])
    converse(["calm songs for studying", "quit"], llm, tmp_path)
    files = os.listdir(tmp_path)
    assert len(files) == 1
    with open(tmp_path / files[0], encoding="utf8") as f:
        log = f.read()
    assert "User: 'calm songs for studying'" in log
    assert "Recommendations: Quiet Library" in log

def test_blocked_input_never_reaches_gemini_or_the_log(tmp_path):
    llm = FakeLLM([CHILL])
    iv, said, prompts = converse(
        ["my email is someone@example.com", "something chill", "quit"], llm, tmp_path)
    assert llm.extract_calls == ["something chill"]          # the email never reached Gemini
    assert "Please don't share personal details" in said
    assert prompts[0] == prompts[1]                          # same question asked again
    log = (tmp_path / os.listdir(tmp_path)[0]).read_text(encoding="utf8")
    assert "someone@example.com" not in log                  # and never reached the log
    assert "Blocked input (category: personal_data)" in log

# ---------- ratings ----------

def start(llm, tmp_path, replies=()):
    """An interview that already knows the user's taste and has shown its picks."""
    said = []
    replies = iter(replies)
    iv = Interview(SONGS, FakeRetriever(), llm, ask_user=lambda prompt: next(replies),
                   say=said.append, log_dir=str(tmp_path),
                   profile={**CHILL, **STUDY, "target_energy": 0.5})
    iv.open_log()
    iv.respond("recommend")
    return iv, said


def test_goodbye_is_said_once_at_the_end(tmp_path):
    _, said, _ = converse(["profile", "profile", "quit"], FakeLLM(), tmp_path)
    assert said.count("Goodbye! Enjoy the music.") == 1


def test_rate_command_collects_ratings_and_updates_the_profile(tmp_path):
    iv, said = start(FakeLLM(), tmp_path, replies=["9", "2"])
    iv.respond("rate")
    assert iv.rated_ids == {1, 2}
    assert len(iv.history) == 1 and iv.history[0]["mean"] == 5.5
    assert iv.profile["target_energy"] != 0.5            # moved by the ratings
    assert any("average 5.5/10" in line for line in said)
    iv.close_log()


def test_rate_asks_again_after_a_bad_number_and_can_be_cancelled(tmp_path):
    iv, said = start(FakeLLM(), tmp_path, replies=["eleven", "0", "cancel"])
    iv.respond("rate")
    assert iv.rated_ids == set() and iv.history == []
    assert sum("Please start with a number" in line for line in said) == 2
    iv.close_log()


def test_rate_comment_is_sent_to_gemini(tmp_path):
    llm = FakeLLM([{"target_energy": 0.1}])
    iv, _ = start(llm, tmp_path, replies=["3 way too loud", "8"])
    iv.respond("rate")
    assert llm.extract_calls and "way too loud" in llm.extract_calls[0]
    assert iv.profile["target_energy"] < 0.5           # moved toward Gemini's 0.1, but not all the way
    iv.close_log()


def test_rate_before_any_recommendation_does_not_crash(tmp_path):
    said = []
    iv = Interview(SONGS, FakeRetriever(), FakeLLM(), say=said.append, log_dir=str(tmp_path))
    iv.respond("rate")
    assert "no songs to rate" in " ".join(said)


def test_submit_ratings_rejects_incomplete_ratings(tmp_path):
    import pytest
    iv, _ = start(FakeLLM(), tmp_path)
    with pytest.raises(ValueError):
        iv.submit_ratings({1: 9})            # song 2 is missing
    iv.close_log()


# ---------- reference for extraction ----------

def test_extraction_reference_always_includes_every_attributes_word_table():
    from src.interview import extraction_reference

    class DocsRetriever(FakeRetriever):
        chunks = [
            ("attributes.md", '## energy\nEnergy is intensity.\nWords to numbers: "quiet" = 0.25'),
            ("attributes.md", '## acousticness\nReal instruments.\nWords to numbers: "acoustic" = 0.8'),
            ("how_it_works.md", "## scoring\nNo table here."),
        ]

    snippets = dict(extraction_reference(DocsRetriever(), {"favorite_mood": {"calm"}}, ["fast"]))
    assert 'energy: "quiet" = 0.25' in snippets["word-to-number tables"]
    assert 'acousticness: "acoustic" = 0.8' in snippets["word-to-number tables"]
    assert "scoring" not in snippets["word-to-number tables"]
    assert "song catalog" in snippets and "attributes.md" in snippets   # catalog labels + retrieved text
