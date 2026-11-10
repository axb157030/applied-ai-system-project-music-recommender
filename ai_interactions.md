# AI interactions: reasoning traces

This file shows how the agent reasons, step by step, using the log the system wrote while running (`logs/interview_20261110_123348.txt`, produced by the same session as `logs/demo_session.txt`). Long lines are cut at 260 characters with `...`. Nothing else is edited.

## How to read the trace

The interview writes one line for every decision it makes. The line types:

| Line starts with | Meaning |
|---|---|
| `User:` | text that passed the input guardrail |
| `Blocked input (category: ...)` | the guardrail stopped the text. Only the category is logged, never the text |
| `Treating as a question` / `RAG answer` | the agent chose the RAG branch and what Gemini answered from the retrieved paragraphs |
| `Unknown attributes: [...]; asking about X` | the **plan** step: which preference do we know least about |
| `Retrieved for X` / `Reference for extraction` | the **retrieve** step: which doc files were given to Gemini |
| `Question:` | the question Gemini wrote from those docs |
| `Extracted:` | what Gemini returned **after** validation (unknown fields and labels already removed) |
| `Profile now:` | the profile after merging |
| `Recommendations:` | the top 5 with scores (from code, not Gemini) |
| `Ratings:` / `Profile before/after:` | the feedback step: metrics, then what the arithmetic and the comment changed |

## The trace

```text
Initial profile: {}
User: 'what is energy?'
Treating as a question: 'what is energy?'
RAG answer: Energy says how intense and active a song feels, ranging from a scale of 0 to 1 where low values are quiet, slow, and gentle, and high values are loud, fast, and driving. For example, "Weightless" by Marconi Union is a low energy song (0.10), wh...

Files relied on: attributes.md, examples_and_listening.md
Unknown attributes: ['favorite_mood', 'preferred_occasions', 'target_energy', 'favorite_genre', 'vocals', 'target_valence', 'target_acousticness', 'language', 'preferred_decade', 'target_popularity', 'target_tempo', 'target_danceability', 'target_duration']...
Retrieved for 'favorite_mood': ['glossary_and_faq.md', 'moods_genres_and_tags.md', 'attributes.md']
Question: What is the single main feeling you want from a song, choosing strictly from calm, chill, energetic, happy, intense, nostalgic, relaxed, romantic, sad, or uplifting?
Blocked input (category: prompt_injection)
User: 'something calm to study to'
Reference for extraction: ['interview_playbook.md', 'attributes.md', 'attributes.md', 'glossary_and_faq.md', 'moods_genres_and_tags.md', 'attributes.md', 'word-to-number tables', 'song catalog']
Extracted: {'favorite_mood': 'calm', 'target_energy': 0.25, 'vocals': 'instrumental', 'preferred_occasions': ['study'], 'target_instrumentalness': 0.95}
Profile now: {'favorite_mood': 'calm', 'target_energy': 0.25, 'vocals': 'instrumental', 'preferred_occasions': ['study'], 'target_instrumentalness': 0.95}
Recommendations: Comptine d'un autre ete (1.00); Sweden (0.99); One Summer's Day (0.97); Cornfield Chase (0.97); Clair de Lune (0.97)
User: 'I like acoustic'
Reference for extraction: ['examples_and_listening.md', 'interview_playbook.md', 'attributes.md', 'word-to-number tables', 'song catalog']
Extracted: {'target_acousticness': 0.8}
Profile now: {'favorite_mood': 'calm', 'target_energy': 0.25, 'vocals': 'instrumental', 'preferred_occasions': ['study'], 'target_instrumentalness': 0.95, 'target_acousticness': 0.8}
Recommendations: Comptine d'un autre ete (0.98); One Summer's Day (0.98); Deltarune - Beginning/A Town Called Hometown (Orchestral Arrangement) (0.97); Sweden (0.96); Undertale - Ruins (Orchestral Remix) (0.96)
User: 'rate'
Ratings: {36: 9, 19: 8, 13: 3, 16: 7, 14: 6}; metrics: {'count': 5, 'mean': 6.6, 'hit_rate': 0.6, 'liked': [36, 19, 16], 'disliked': [13]}
Profile before: {'favorite_mood': 'calm', 'target_energy': 0.25, 'vocals': 'instrumental', 'preferred_occasions': ['study'], 'target_instrumentalness': 0.95, 'target_acousticness': 0.8}
Profile after:  {'favorite_mood': 'calm', 'target_energy': 0.537, 'vocals': 'instrumental', 'preferred_occasions': ['study'], 'target_instrumentalness': 0.949, 'target_acousticness': 0.816, 'target_tempo': 150.0}
Comment "too slow and sleepy" changed: target_energy, target_tempo.
Recommendations: Undertale - Once Upon A Time (Orchestral Cover) (0.93); Cornfield Chase (0.90); Clair de Lune (0.81); Weightless (0.70); Take Five (0.64)
User: 'quit'
```

## What this trace demonstrates

1. **Multi-step reasoning (agentic).** Plan (`Unknown attributes`, ask about the first), retrieve (`Retrieved for`), act (Gemini writes the question), observe (the user's reply), check (`Extracted` is validated), update (`Profile now`), and stop asking once two preferences are known.
2. **Branching.** `what is energy?` takes the RAG branch and then the same question is asked again. The injection attempt is stopped before any Gemini call and the interview continues unchanged.
3. **Grounding.** The RAG answer names the files it used. The recommendations come from the catalog only.
4. **Feedback.** The `Profile before/after` lines show exactly which numbers the ratings and the comment moved. The ratings (9, 8, 3, 7, 6) averaged 6.6 with a 60% hit rate, and the comment on the song rated 3 moved the energy target up (0.25 to 0.537) because the listener said "too slow".

## Other saved traces

- [`logs/eval_results.txt`](logs/eval_results.txt): the evaluation harness run (73 checks).
- [`logs/rag_and_extraction_check.txt`](logs/rag_and_extraction_check.txt): the same questions answered with and without the docs, and free-text extraction examples.
- [`logs/feedback_round_1.txt`](logs/feedback_round_1.txt) and [`logs/end_to_end_check.txt`](logs/end_to_end_check.txt): earlier live runs from before the feedback rule and the extraction reference were fixed. Kept as evidence of the bugs described in the README.
