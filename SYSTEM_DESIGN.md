# System Design: how it works and why

The README says what the project does. This file explains how the parts fit together and why each choice was made.

---

## 1. The core idea: split the work by what each part is good at

| Job | Tool | Why |
|---|---|---|
| Understand messy language ("something chill but not sleepy") | Gemini | Language is what LLMs do well |
| Explain a concept in plain words | Gemini + retrieved docs (RAG) | Fluent, but limited to our own text |
| Rank songs | Deterministic code (`recommender.py`) | Repeatable, testable, explainable, cannot invent songs |
| Check facts (is this label in the catalog? is energy between 0 and 1?) | Code (`preferences.py`) | Never trust an LLM to verify itself |
| Update numbers from ratings | Code (`feedback.py`) | It is arithmetic. Exact, free, same result every time |
| Read a written comment ("too slow") | Gemini, then checked by code | Code cannot read language, but Gemini's answer can be wrong |

An LLM is a function from text to text. It is not a database or a calculator. The system lets it do language work and lets code do fact work.

---

## 2. The agent loop (`interview.py`)

Each turn of the conversation:

```
show a prompt                  opening question, or a question about the attribute we know least about
read what the user typed
  input check (guardrails)     blocked? show a safe reply, log only the category, continue
  command?                     quit, recommend, rate, profile, skip, refine, reset
  ends with "?"                RAG: retrieve doc paragraphs, Gemini answers only from them, then re-ask
  otherwise                    extract_preferences (Gemini, JSON) -> validate -> merge into the profile
once the profile has 2+ preferences: recommend, show the reasons, keep refining
"rate": collect 1-10 ratings (+ comments) -> feedback.py -> new profile -> new unseen songs
```

This is an agent in the plain sense: the system **plans** (which attribute is unknown?), **acts** with tools (retriever, Gemini, validator, recommender), **checks** the result, and repeats. It is bounded: at most 6 questions, at most 3 side questions per question, and `skip` always works. Every decision is written to `logs/interview_<time>.txt`. Those files are the reasoning traces (see `ai_interactions.md`).

The interview never uses RAG to decide the user's preferences. RAG answers the user's questions. Preferences come only from what the user says.

---

## 3. RAG (`retriever.py`, `docs/`)

1. **Load** every `.md` file in `docs/`.
2. **Chunk** by blank line, so each paragraph (one attribute, one label list) is a chunk.
3. **Index**: an inverted index from each word (lowercased, plural-stemmed, filler words removed) to the chunks that contain it.
4. **Retrieve**: score chunks by how many query words they contain (a heading match counts extra). A question that matches nothing gets **no chunks**, and the app answers "I do not know based on the docs" without calling Gemini.
5. **Generate**: Gemini gets only the retrieved chunks and the rule "use only these snippets or say you don't know".

Three sources feed the model: the retrieved paragraphs (for questions), the word-to-number table of every attribute (so "acoustic" always becomes 0.8), and the labels that really exist in the catalog (so it cannot invent a mood).

Measured effect (`logs/rag_and_extraction_check.txt`): asked "How does the app handle explicit songs?", Gemini without docs invented "[E] badges" and an "Explicit Content Filter". With RAG it described the real behavior (a filter, applied before ranking). Asked "What is the weather in Paris?" the retriever returned nothing and the app refused.

Limit: keyword matching. A question phrased with none of the doc's words finds nothing. Embeddings are the natural upgrade.

---

## 4. Guardrails and reliability

| Risk | Defense | File |
|---|---|---|
| Prompt injection ("ignore your instructions") | Pattern check on the user's text before Gemini sees it | `guardrails.py` |
| Personal data (email, phone, card number with checksum) | Blocked, and never written to the logs | `guardrails.py` |
| Self-harm | A caring reply with a crisis line, not a refusal | `guardrails.py` |
| Huge input | 500-character limit | `guardrails.py` |
| Gemini returns bad JSON, a code fence, or invented fields | Strip fences, keep only known fields, clamp ranges, drop labels not in the catalog | `preferences.py` |
| Invented song | Impossible: only catalog rows can be returned | `recommender.py` |
| Gemini down or quota exceeded | Retry once, raise `LLMError` (never returned as text); the interview falls back to fixed questions | `llm.py`, `interview.py` |
| Invalid ratings (0, 11, "ten", `True`, missing songs) | Rejected with a clear error; Regenerate stays locked until all songs are rated | `feedback.py`, `app.py` |

Why no long word blocklist: it can't tell "I hate sad songs" from abuse, and in a music app the false alarms are the common case.

---

## 5. Feedback (`feedback.py`)

**Measuring.** Mean rating, and hit rate (share of songs rated 7+). Note that a match percentage is not satisfaction. In testing a 96% match was rated 2/10.

**Improving.** For every numeric target in the profile:

```
weight of a song = its rating - the round's average rating
target = target + 0.5 * sum(weight * (song value - target)) / sum(|weight|)
```

Songs rated above the round's average pull the target toward them, and songs below it push away. Using the round's own average means there is always a direction when the ratings differ, even if every song was rated low. (An earlier "liked 7+ / disliked 4-" rule had no direction when everything was disliked and failed the simulated-user test.) Equal ratings carry no information, so nothing changes. Labels (mood, genre, occasions) are not changed by ratings.

**Comments.** Each comment goes to `extract_preferences` with the song and its rating ("rated it 3/10, comment: too slow, give the preferences they want INSTEAD"). The result is validated, and numeric targets move only halfway toward it because Gemini's numbers vary between runs.

---

## 6. Evaluation (`eval/run_eval.py`)

Offline (deterministic, free): guardrail cases, retrieval answer/refuse cases, six personas (gym, study, sleep, party, clean hip hop, instrumental) with checks on their top 5, properties that must hold for any profile (variety, real ids, sorted scores, empty profile), and a simulated user with a hidden taste rating three rounds. `--live` adds Gemini checks: free text to profile to top 5, and a real RAG answer and refusal.

A simulated user is cheap and repeatable, and it found a real bug (section 5). Its weakness: the simulated taste is one number, so passing proves the loop can learn a simple taste, not that real people will be satisfied.

---

## 7. Known limits and next steps

- Ratings change only numeric targets, never mood, genre or occasion, which carry most of the score. After several rounds the list can stay mood-bound.
- Gemini's suggested numbers differ between runs even at temperature 0, which is why comments are blended and not trusted.
- Catalog values (energy, valence...) were estimated by hand, 50 songs only, content-based only (no "people like you" signal).
- Next steps: embeddings for retrieval, an LLM judge scoring explanations (with human spot checks), learning labels from ratings, a larger catalog.
