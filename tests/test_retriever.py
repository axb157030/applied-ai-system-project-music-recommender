import pytest
from src.retriever import Retriever  # adjust if your file lives elsewhere


@pytest.fixture
def docs_dir(tmp_path):
    """A small fake docs/ folder. tmp_path is a built-in pytest fixture that
    gives each test its own temporary directory, deleted afterward."""
    (tmp_path / "attributes.md").write_text(
        "mood: a label like happy or chill.\n\n"
        "energy: a number from 0 to 1.",
        encoding="utf8",
    )
    (tmp_path / "how_it_works.md").write_text(
        "Each song gets a score based on its mood and energy.",
        encoding="utf8",
    )
    (tmp_path / "ignored.csv").write_text("not a doc", encoding="utf8")
    return tmp_path


@pytest.fixture
def retriever(docs_dir):
    return Retriever(docs_folder=str(docs_dir))


class FakeLLM:
    """Stands in for GeminiClient so tests never hit the network."""

    def __init__(self):
        self.calls = []

    def answer_from_snippets(self, query, snippets):
        self.calls.append((query, snippets))
        return "fake answer"


# ---------- loading and chunking ----------

def test_loads_only_md_and_txt_files(retriever):
    filenames = sorted(name for name, _ in retriever.documents)
    assert filenames == ["attributes.md", "how_it_works.md"]


def test_missing_folder_gives_no_documents(tmp_path, capsys):
    r = Retriever(docs_folder=str(tmp_path / "does_not_exist"))
    assert r.documents == []
    assert r.chunks == []
    assert "Warning" in capsys.readouterr().out


def test_documents_split_into_paragraph_chunks(retriever):
    # attributes.md has 2 paragraphs, how_it_works.md has 1
    assert len(retriever.chunks) == 3


# ---------- index ----------

def test_index_maps_word_to_chunk_ids(retriever):
    chunk_ids = retriever.index["mood"]
    files = {retriever.chunks[i][0] for i in chunk_ids}
    # "mood" appears in one chunk of each file
    assert files == {"attributes.md", "how_it_works.md"}


def test_index_values_are_sets_without_duplicates(retriever):
    assert isinstance(retriever.index["mood"], set)


# ---------- scoring and retrieval ----------

def test_repeated_query_word_not_double_counted(retriever):
    assert retriever.score_document("mood mood", "the mood of a song") == 1


def test_retrieve_returns_most_relevant_chunk_first(retriever):
    results = retriever.retrieve("how is the mood score calculated")
    assert results[0][0] == "how_it_works.md"


def test_single_keyword_question_is_answered(retriever):
    assert retriever.retrieve("What is energy?") != []


def test_off_topic_question_is_refused(retriever):
    assert retriever.retrieve("payment processing") == []


def test_stopwords_only_query_returns_nothing(retriever):
    assert retriever.retrieve("what is the") == []


def test_top_k_limits_results(retriever):
    assert len(retriever.retrieve("mood energy", top_k=1)) == 1


# ---------- answer modes ----------

def test_retrieval_only_includes_filename_and_text(retriever):
    answer = retriever.answer_retrieval_only("mood score")
    assert "[how_it_works.md]" in answer
    assert "score based on its mood" in answer


def test_retrieval_only_refuses_off_topic(retriever):
    answer = retriever.answer_retrieval_only("payment processing")
    assert answer == "I do not know based on these docs."


def test_rag_requires_llm_client(retriever):
    with pytest.raises(RuntimeError, match="requires an LLM client"):
        retriever.answer_rag("mood score")


def test_rag_passes_snippets_to_llm(docs_dir):
    llm = FakeLLM()
    r = Retriever(docs_folder=str(docs_dir), llm_client=llm)

    answer = r.answer_rag("mood score")

    assert answer == "fake answer"
    assert len(llm.calls) == 1
    query, snippets = llm.calls[0]
    assert query == "mood score"
    assert snippets  # non-empty list of (filename, text)


def test_rag_refusal_does_not_call_llm(docs_dir):
    llm = FakeLLM()
    r = Retriever(docs_folder=str(docs_dir), llm_client=llm)

    answer = r.answer_rag("payment processing")

    assert answer == "I do not know based on these docs."
    assert llm.calls == []  # LLM was never called

# ---------- behaviors added after testing on the real docs ----------

def test_filler_words_do_not_block_an_answer(retriever):
    # "mean" is question filler, so this is searched as just "energy"
    assert retriever.retrieve("what does energy mean") != []


def test_plural_question_word_matches_singular_doc_word(retriever):
    # docs say "score", the question says "scores"
    assert retriever.retrieve("scores") != []


def test_heading_match_ranks_first(tmp_path):
    (tmp_path / "a.md").write_text(
        "## other\nenergy is mentioned here in passing.\n\n## energy\nEnergy means intensity.",
        encoding="utf8",
    )
    r = Retriever(docs_folder=str(tmp_path))
    assert r.retrieve("what is energy")[0][1].startswith("## energy")


def test_long_off_topic_question_is_refused(retriever):
    # "mood" matches one chunk, but a 3-keyword question needs 2 matching words
    assert retriever.retrieve("weather forecast mood") == []
