# System Design: how it works, why, and how to build it in about 4 hours

This is the in-depth companion to the README. It answers your design questions, then gives a build order where each step teaches one AI foundation.

---

## 1. The core idea: split the work by what each part is good at

| Job | Best tool | Why |
|---|---|---|
| Understand messy human language ("something chill but not sleepy") | LLM (Gemini) | Language understanding is what LLMs do well |
| Explain a concept in plain words | LLM + retrieved docs (RAG) | Fluent, but grounded in your text |
| Rank songs by match | Deterministic code (your Module 1-3 scorer) | Repeatable, testable, explainable, can't hallucinate |
| Decide what's *true* (does the song exist? is energy really 0.82?) | Code checks (guardrails) | Never trust an LLM to verify itself on facts |
| Judge quality of explanations | A second LLM call ("judge") | Subjective, so an LLM is acceptable, but verified by humans |

**Foundation #1 to learn:** an LLM is a function from text to text. It is not a database or a calculator. Build systems where it does language work and code does fact work.

---

## 2. Your questions, answered

### What external source should the RAG use?
**A small folder of `.md` files in `docs/`, plus the song catalog as a second source.** The DocuBot example works exactly this way: split documents into chunks, find the chunks most relevant to the question, and give only those to the LLM.

Yes, the docs should tell the agent how to interpret users and when to ask more. Three kinds of document:

| File | Contents | Used when |
|---|---|---|
| `docs/attributes.md` | For every attribute: a plain-English definition, scale anchors ("0.1 = solo piano, 0.5 = Fleetwood Mac, 0.9 = metal"), and a **word-to-number table** ("high energy" -> 0.8-0.9, "kind of low" -> 0.3) | Asking a question, explaining jargon, parsing "a high amount of energy" |
| `docs/genres.md` | Each genre: a one-line description and 2-3 well-known example songs/artists, plus related genres | User asks about genres, or says "I like Khruangbin" (map artist to genre/mood) |
| `docs/interview_playbook.md` | Rules: ask about broad attributes first (mood, energy, genre), ask at most 2 questions per attribute, stop when confident, never ask what the user already told you, what to do when the user says "I don't know" (offer 3 concrete choices) | Deciding *what to ask next* and *when to stop* |

Format: **Markdown.** It's readable by you, by Gemini, and in git diffs. Use one `##` heading per chunk so chunking is trivial (split on headings).

**Foundation #2 (RAG):** retrieve first, then generate from what you retrieved. It reduces hallucination and lets you change the AI's knowledge by editing a file, with no retraining.

### Should the app keep asking random questions until the user stops?
No. Random questions waste the user's time and aren't testable. Use a **bounded, adaptive interview**:

- Keep a **confidence** per attribute (0 = unknown, 1 = sure).
- Each turn, the agent **plans**: pick the attribute with the lowest confidence that matters most (weights from your scorer tell you which).
- Ask about it. Parse the answer. Update value and confidence.
- **Stop** when: every important attribute has confidence >= 0.7, *or* max turns (suggest 8), *or* the user presses "Stop, show me songs".
- Always keep a **Stop** button visible. The user is in control.

### One question per attribute, or several?
Group your ~15 attributes into **5-6 question topics**, not 15 questions:

1. Vibe/mood and occasion ("studying, workout, driving, sad night in?") -> mood, mood tags, valence
2. Energy and pace -> energy, tempo, danceability
3. Sound texture -> acousticness, instrumentalness, vocals vs instrumental
4. Genres and artists you know -> genre, similar-artist seeds
5. Era and familiarity -> decade, popularity (mainstream vs hidden gems)
6. Practical -> duration, language, explicit content OK?

A user's single answer often fills several attributes. ("Late-night studying, no lyrics" fills mood, tags, vocals, and energy). That's the real value of an LLM here: **extract many structured fields from one free-text answer**.

### The user asks "what is energy?"
Handle it as a branch in the loop. If the answer is a question or confusion ("what's that?"), the agent retrieves `docs/attributes.md#energy`, Gemini explains it in 2-3 sentences with anchors, then re-asks. This is the best demo of RAG in the project.

### Slider with audio / genre snippets?
Skip audio for this project. It needs licensed clips or an external API, hosting, and playback code, and it adds no new AI concepts. **Do instead:**

- **Word anchors**: "Low energy (0.2): *Weightless* by Marconi Union. High energy (0.9): *Killing in the Name* by Rage Against the Machine."
- **Genre picker**: show a list of genre chips (from `genres.md`) with a one-line description and example artists, either up front or when the user asks. Add the multiselect `st.pills`/`st.multiselect`.
- **Slider** in the Profile tab for manual control (0-1), which is the "tab allowing users to modify attributes" you wanted.
- Future work: link out to a YouTube search for each example.

### Do users say numbers or words?
Mostly words. The parser's job is to map words to numbers using the table in `attributes.md`. Ask Gemini for **JSON only**, with a schema, e.g. `{"energy": {"value": 0.85, "confidence": 0.8}}`. Then **validate in code**: clamp to range, drop unknown keys, reject malformed JSON and retry once.

### Recommended expanded attributes
Keep what you have, and add (all 0-1 unless noted): `instrumentalness`, `speechiness`, `liveness`, `loudness_norm`, `language` (text), `explicit` (bool), `era` already as `release_decade`, `lyrical_themes` (tags), `similar_artists` (user seed list), `occasion_tags` ("gym|study|road trip"). Populate `data/songs.csv` with 40-60 songs. Gemini can help you draft rows, **but you must sanity-check the values yourself**; it will guess.

---

## 3. How the agent loop works (Foundation #3: agentic workflow)

```
state = UserProfile(all confidence 0), history = []
while not done:
    target   = choose_attribute(state)                # PLAN   (code)
    context  = retriever.search(target.name)          # RETRIEVE (RAG)
    question = gemini.ask(context, history, target)   # ACT    (LLM)
    answer   = ui.get_user_input()                    #        (human)
    if guardrails.block(answer): continue             # CHECK  (code)
    if is_question(answer): explain(context); continue
    updates  = gemini.extract_json(answer, schema)    # ACT    (LLM)
    state.apply(validate(updates))                    # CHECK  (code)
    done = confident(state) or turns >= MAX or stop_pressed
```

An *agent* is just this loop: the system plans, acts using tools (retriever, LLM, parser), checks the result, and repeats. Log each iteration to `logs/*.jsonl`. These are your **reasoning traces**. Copy one into `ai_interactions.md` for the +2 bonus.

---

## 4. How to know the AI performed well (evaluation)

Your idea (ratings after each round) is the right foundation. Make it stronger by measuring several things:

| Metric | How | What it tells you |
|---|---|---|
| **Mean rating** | Average of the five 1-10 sliders | Overall satisfaction |
| **Hit rate** | Share of songs rated >= 7 ("precision@5") | How many picks were actually good |
| **Round-over-round change** | Mean rating in round 2 minus round 1 | Did using feedback help? This is your headline result |
| **Rank agreement** | Do high-scored songs get higher ratings? (Spearman correlation between system rank and user rating) | Is the *ordering* meaningful? |
| **Interview efficiency** | Questions asked until stop | Fewer is better at equal quality |
| **Guardrail stats** | Blocked inputs, regenerated explanations, fallbacks | How often the safety nets fire |
| **LLM-as-judge score** | A second Gemini call grades each explanation (accurate? specific? mentions real values?) on 1-5 with a reason | Cheap, automatic quality signal on text |
| **Persona harness** | Fake users with known tastes (e.g. "gym person"), where the answer is checkable in code | Repeatable testing without humans |

Better than asking only humans: **simulated personas**. Write 5-8 personas as dicts (their true preferences). A scripted "user" answers the agent's questions from those preferences. Check automatically that the top 5 match (average energy near the target, genre matches, and so on). Pass/fail results go into the README table. This is the **test harness** bonus.

**Foundation #4:** you can't improve what you can't measure. Be honest about the limits too: a 1-10 slider is subjective and noisy (people rate the first song higher, and 5 songs is a tiny sample), and an LLM judge has its own biases (it tends to like fluent text). Say so in `model_card.md`.

**Profile update from ratings (simple and explainable):**
```
liked    = songs rated >= 7;  disliked = songs rated <= 4
for each numeric attribute a:
    target[a] += lr * (mean(liked[a]) - target[a])       # move toward liked
    target[a] -= lr * (mean(disliked[a]) - target[a])    # move away from disliked
clamp to [0, 1]   # lr ~ 0.3
```
Optional comments get passed to Gemini with the extractor schema, so "too slow" becomes a tempo/energy adjustment. Unlock the Regenerate button only once all 5 are rated, as you proposed. This guarantees every round gives signal.

---

## 5. Guardrails

| Risk | Guardrail | Type |
|---|---|---|
| Prompt injection ("ignore your rules") | Pattern + length check; the user text is placed in a delimited block and the system prompt says to treat it as data | Input |
| Out-of-range / nonsense values | Clamp to 0-1, drop unknown keys | Input/parse |
| Malformed JSON from Gemini | Retry once, then re-ask the user | Parse |
| Hallucinated song | Songs come only from the scorer; explanation check verifies song ID and title exist | Output |
| Explanation cites wrong numbers | Compare every number in the text to the song row | Output |
| API down, timeout, rate limit | Retry with backoff, then **fallback to template text** from the original `reasons` list. The app still works without Gemini | Reliability |
| Secrets leak | `.env`, never logged, never in the repo | Security |

---

## 6. Module layout (single responsibility each)

| File | Responsibility | Tested by |
|---|---|---|
| `src/models.py` | `Song`, `UserProfile` (expanded, value + confidence), `Rating` dataclasses | unit |
| `src/catalog.py` | Load/validate `songs.csv` | unit |
| `src/recommender.py` | Port of your Module 1-3 scorer, extended to new attributes | unit |
| `src/retriever.py` | Chunk `docs/*.md` by heading, score by keyword overlap, return top-k chunks with their source | unit |
| `src/llm.py` | One wrapper around Gemini: key from env, timeout, retry, JSON mode, logging | mocked unit |
| `src/guardrails.py` | Input and output checks | unit |
| `src/interview.py` | The agent loop (Section 3) | harness |
| `src/feedback.py` | Ratings -> metrics, profile update | unit |
| `src/evaluator.py` | LLM judge | harness |
| `src/app.py` | Streamlit tabs: Interview, Profile, Recommendations & Ratings | manual |
| `eval/run_eval.py` | Personas -> summary table | itself |

Putting all Gemini calls behind `llm.py` means you can swap in a **fake LLM** in tests. Tests then run free, offline, and deterministically.

---

## 7. Build order (about 4 hours), with the lesson for each step

| Time | Build | You'll learn |
|---|---|---|
| 0:00-0:30 | Port the scorer. Add 4-5 new attributes. Write `songs.csv` (start with 30 songs). Write 3 unit tests. Get a CLI printing top 5. | The deterministic baseline comes first. You can't judge AI against nothing. |
| 0:30-1:00 | `llm.py` + one call: "extract energy from 'I want something calm'" returning JSON. Validate it. | Prompting for structured output, and why validation is mandatory |
| 1:00-1:45 | Write the three docs. Build `retriever.py` (split on `##`, keyword overlap). Test: query "what is energy" returns the energy chunk. | RAG from scratch with no library, so it stops being magic |
| 1:45-2:30 | `interview.py` loop + guardrails + CLI chat. Log traces. | Agentic loop: plan, act, check |
| 2:30-3:15 | `feedback.py` metrics + profile update, then Streamlit app (3 tabs). | Closing the loop, and evaluation metrics |
| 3:15-3:45 | `eval/run_eval.py` with 5 personas, plus the LLM judge. Run, paste real output into the README. | Testing AI systems honestly |
| 3:45-4:00+ | `model_card.md`, `ai_interactions.md`, final diagram check. | Responsible-AI reflection |

**Cut line if you're behind:** drop the LLM judge and the Streamlit styling; keep CLI + RAG + loop + harness. Everything required by the rubric still holds.

---

## 8. Keeping this *your* work (and graded honestly)

- The rubric asks for **one AI suggestion you accepted and one you rejected or changed**. Keep a running note in `ai_interactions.md` as you go, because it's hard to remember later. Candidates already visible in this design: audio snippets (suggested, scoped out), embeddings (deferred to keyword retrieval), the 15-question interview (grouped into 6 topics).
- Write each module yourself and ask the AI to explain or review it. Don't ask it to write the module. If you can explain to a friend why `llm.py` retries and why `interview.py` has a max-turn cap, you've learned it.
- Never paste output into the README that you didn't produce by running the code.
- `model_card.md` is written last, from what actually happened.
