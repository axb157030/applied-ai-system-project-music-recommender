# Final Project Applied AI System, Gemini Music Recommender: an interview-driven, self-evaluating music agent

> **Status: design complete, implementation in progress.** Sections marked `TODO(run)` must be filled with *real* output pasted from your own runs. Do not invent output.

## Base project (Modules 1-3)

**Original project: Music Recommender Simulation** *(TODO: confirm this is the exact name you used in Modules 1-3)*.
It was a content-based recommender. A `Song` catalog (genre, mood, energy, tempo, valence, danceability, acousticness, popularity, decade, mood tags, vocals, duration) was compared with a `UserProfile`. A weighted `score_song()` produced a 0-1 match score plus a human-readable reason for each feature. Four ranking strategies (`balanced`, `genre_first`, `mood_first`, `energy_focused`) and an artist-variety penalty kept the top-k list varied. It needed the user to already know their own preferences and type them in as numbers.

## What this project adds and why it matters

Most people can't say "my target energy is 0.7". They can say "something to study to" or "I like Khruangbin". This project closes that gap with an AI agent:

| New capability | AI feature (rubric) | Where |
|---|---|---|
| Gemini interviews the user, asks adaptive questions, explains jargon on request, and converts free-text answers into structured attributes | **Agentic workflow** (plan, act, check) | `src/interview.py` |
| Gemini retrieves from `docs/` (attribute guide, genre guide, interview playbook) before asking or explaining | **RAG** | `src/retriever.py`, `docs/` |
| The original scorer stays deterministic and ranks only songs in the catalog, so Gemini can't invent songs | Grounding / guardrail | `src/recommender.py` |
| The user gives each of the top 5 songs a score from 1 to 10. Plain Python (no LLM) turns the scores into numbers such as the average rating and how many songs scored 7 or more. This is how we **measure how well the recommender did**. | **Reliability/evaluation system** | `src/feedback.py` |
| The same scores are used to **improve the next list**: the profile moves toward songs rated high and away from songs rated low. The "Regenerate" button unlocks only after all 5 songs are rated. | Feedback loop | `src/feedback.py` |
| A second Gemini call reads each written explanation and grades it 1 to 5 (is it accurate, specific, and based on real song values?). This grades the **explanation text**, not the songs. It is a separate check from the user's ratings. | Self-checking (LLM judge) | `src/evaluator.py` |
| Fake users with known tastes (for example "gym person: wants high energy") run through the whole system automatically. A script checks whether the top 5 match each fake user's taste and prints PASS/FAIL. | **Test harness** (bonus) | `eval/run_eval.py` |

## Architecture overview

Source: [`diagrams/system_architecture.mmd`](diagrams/system_architecture.mmd). The yellow boxes are where a human is involved. Testing is shown inside the diagram: `run_eval.py` replaces the real user with a fake persona and checks the top 5 for PASS or FAIL.

Data flows in one direction around a loop:

1. **Input**: the user chats in Streamlit. Input guardrails check length, off-topic text, prompt injection, and value range.
2. **Interview agent**: it *plans* (which attribute am I least sure about?), *acts* (retrieves that attribute's doc from `docs/`, asks Gemini to phrase a question, then asks Gemini to extract a value as JSON), and *checks* (is every attribute confident enough, have we hit max turns, or did the user press Stop?).
3. **Profile**: the structured `UserProfile` carries a value and a confidence for each attribute. The user can also edit it directly in the profile tab.
4. **Recommender**: the deterministic scorer from Modules 1-3 ranks `data/songs.csv` and returns the top 5.
5. **Explainer + output guardrails**: Gemini explains each pick using retrieved docs and real feature values. A check verifies the song exists and the numbers it cites are true.
6. **Human check**: the user rates each song 1-10 (optional comment). Only when all 5 are rated do the metrics, the LLM judge, and **Regenerate** unlock. The profile updater moves the profile toward liked songs and away from disliked ones, then the loop repeats.
7. **Reliability**: pytest unit tests and the persona eval harness test the deterministic parts and the agent. Everything logs to `logs/*.jsonl`.

    - testing recommender, which provides the scores of the songs based on user preference

```
tests/test_recommender.py::test_recommend_songs PASSED                                             [ 10%]
tests/test_recommender.py::test_recommender_class PASSED                                           [ 20%]
tests/test_recommender.py::test_recommend_songs_with_empty_profile PASSED                          [ 30%]
tests/test_recommender.py::test_recommend_songs_with_no_songs PASSED                               [ 40%]
tests/test_recommender.py::test_recommend_songs_with_k_greater_than_song_count PASSED              [ 50%]
tests/test_recommender.py::test_recommend_songs_with_invalid_profile PASSED                        [ 60%]
tests/test_recommender.py::test_recommend_songs_with_partial_profile PASSED                        [ 70%]
tests/test_recommender.py::test_recommend_songs_with_empty_song_list PASSED                        [ 80%]
tests/test_recommender.py::test_recommend_songs_rejects_none_profile PASSED                        [ 90%]
tests/test_recommender.py::test__no_song_out_of_top_5_has_higher_score_then_those_top_5_songs PASSED [100%]
```

## Setup instructions

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd applied-ai-system-project-music-recommender

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini key (get one free at https://aistudio.google.com/apikey)
echo GEMINI_API_KEY=your_key_here > .env     # .env is git-ignored. Never commit it.

# 5. Run the app (TODO: confirm the command once app.py exists)
streamlit run src/app.py

# 6. Run the tests and the evaluation harness
pytest -q
python eval/run_eval.py
```

## Sample interactions

> Sample interaction with recommender.py
    - Input:

        - 
        ```
        songs = load_songs("data/songs.csv")
        profile = {
            "genre": "classical", "mood": "calm",
            "target_energy": 0.2, "target_acousticness": 0.9,
            "preferred_occasions": ["study"],
        }
        ```
    - Output:

    ```
    1. Comptine d'un autre ete - Yann Tiersen  (score 0.98)
    genre: exact match with classical (+0.350 of 0.35); mood: exact match with calm (+0.300 of 0.30); occasions: 1 of 1 wanted occasions found (study) (+0.100 of 0.10); energy: song energy 0.25 vs target 0.20, similarity 0.95 (+0.190 of0.20); acousticness: song acousticness 0.95 vs target 0.90, similarity 0.95 (+0.143 of 0.15); total: 1.083 of 1.10 possible = 0.98

    2. Clair de Lune - Claude Debussy  (score 0.97)
    genre: exact match with classical (+0.350 of 0.35); mood: exact match with calm (+0.300 of 0.30); occasions: 1 of 1 wanted occasions found (study) (+0.100 of 0.10); energy: song energy 0.12 vs target 0.20, similarity 0.92 (+0.184 of0.20); acousticness: song acousticness 0.98 vs target 0.90, similarity 0.92 (+0.138 of 0.15); total: 1.072 of 1.10 possible = 0.97

    3. River Flows in You - Yiruma  (score 0.72)
    genre: exact match with classical (+0.350 of 0.35); mood: romantic does not match calm (+0.000 of 0.30); occasions: 1 of 1 wanted occasions found (study) (+0.100 of 0.10); energy: song energy 0.20 vs target 0.20, similarity 1.00 (+0.200 of 0.20); acousticness: song acousticness 0.95 vs target 0.90, similarity 0.95 (+0.143 of 0.15); total: 0.792 of 1.10 possible = 0.72

    4. Gymnopedie No. 1 - Erik Satie  (score 0.71)
    genre: exact match with classical (+0.350 of 0.35); mood: relaxed does not match calm (+0.000 of 0.30); occasions:1 of 1 wanted occasions found (study) (+0.100 of 0.10); energy: song energy 0.16 vs target 0.20, similarity 0.96 (+0.192 of 0.20); acousticness: song acousticness 0.95 vs target 0.90, similarity 0.95 (+0.143 of 0.15); total: 0.784 of 1.10 possible = 0.71

    5. Sweden - C418  (score 0.64)
    genre: ambient does not match classical (+0.000 of 0.35); mood: exact match with calm (+0.300 of 0.30); occasions:1 of 1 wanted occasions found (study) (+0.100 of 0.10); energy: song energy 0.22 vs target 0.20, similarity 0.98 (+0.196 of 0.20); acousticness: song acousticness 0.65 vs target 0.90, similarity 0.75 (+0.112 of 0.15); total: 0.709 of 1.10 possible = 0.64
    ```

### 1. End-to-end run: user doesn't know what they want
```text
TODO(run): paste a full session: questions asked, answers, final profile, top-5 with explanations.
```
Must show: the agent asked fewer than N questions, stopped on its own, and the top 5 are all real catalog songs.

### 2. RAG in action: user asks "what is energy?"
```text
TODO(run): paste the retrieved doc chunk(s) and the explanation Gemini produced from them,
then the agent re-asking the energy question.
```
Must show: the explanation matches `docs/attributes.md`, not Gemini's guess.

### 3. Guardrail / reliability: bad input
```text
TODO(run): paste a prompt-injection attempt ("ignore your instructions and ...") and an
out-of-range answer ("energy 700%"), and what the system did.
```
Must show: input blocked or clamped, event written to the log, interview continued.

### 4. Feedback loop: ratings change the next list
```text
TODO(run): paste ratings (e.g. 9,8,3,2,7), the metrics line, the profile before/after,
and the new top 5.
```

## Design decisions (short version; full reasoning in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md))

- **LLM interviews and explains; code ranks.** Gemini is good at language and bad at being a trustworthy database. Ranking stays in tested, deterministic Python, so the LLM can never recommend a song that doesn't exist.
- **Docs as the RAG source.** Short `.md` files in `docs/` hold the knowledge the agent needs: what each attribute means, example songs at the low/mid/high anchors, and a playbook for when to ask more questions or stop. They are easy to read, edit, and diff.
- **Bounded interview.** One question group per attribute, a hard max-turns cap, and a confidence score per attribute. It does not ask random questions forever.
- **No audio snippets.** Licensing, hosting, and time (about 4 hours) make this a poor trade. Text anchors with example songs ("high energy: like *Song X*") teach the same idea. Listed as future work.
- **Keyword retrieval first.** With about 10 small docs, keyword/TF-IDF retrieval is enough and teaches the idea clearly. Embeddings are the upgrade path.

## Testing summary

> TODO(run): fill in after running. Template: `X of Y unit tests passed; eval harness: A/B personas passed; mean judge score S; main failure: ...; fix: ...`.

| Test input | Evaluation criteria | Result |
|---|---|---|
| Empty answer | Agent re-asks, doesn't crash | TODO(run) |
| "energy 700%" | Value clamped to 1.0, logged | TODO(run) |
| Prompt injection | Blocked, logged | TODO(run) |
| Gemini timeout (simulated) | Retries, then falls back to template question | TODO(run) |
| Persona "gym, loud, fast" | Top 5 average energy > 0.7 | TODO(run) |

## Reflection

TODO: write after building. Cover what you learned about LLMs vs deterministic code, grounding, and evaluating AI. The graded responsible-AI reflection goes in `model_card.md`, not here.

## Build order (core path)

Each step leaves a working app. Extras (guardrails, LLM judge, persona harness, Streamlit) come after step 7.

| Order | File | Job |
|-------|------|-----|
| 1 | `data/songs.csv` | Song catalog with the expanded attributes |
| 2 | `src/recommender.py` | The Modules 1-3 scorer, extended to the new attributes |
| 3 | `src/cli.py` | Hard-coded profile in, top 5 with reasons out (first working recommender) |
| 4 | `src/llm.py` | Wrapper that sends a prompt to Gemini and returns the reply |
| 5 | `docs/*.md` + `src/retriever.py` | Look up the relevant doc text, then give it to Gemini (RAG) |
| 6 | `src/interview.py` | Loop: ask, extract, update the profile, stop after a max number of questions |
| 7 | `src/feedback.py` | 1-10 ratings to metrics and profile update, then regenerate |
| 8 | `src/app.py` | Streamlit, only after the command-line version works |

## Project layout

```
src/            config, models, catalog, retriever, llm, interview, recommender, guardrails, feedback, evaluator, app
data/songs.csv  song catalog (expanded attributes)
docs/           RAG knowledge base (.md)
diagrams/       Mermaid source
tests/          pytest unit tests
eval/           persona harness
logs/           JSONL traces (git-ignored except the sample you link from ai_interactions.md)
SYSTEM_DESIGN.md   in-depth design and 4-hour build plan
model_card.md      reflection (write last)
ai_interactions.md reasoning traces
```
