# Glossary and frequently asked questions

## what is a recommender
A recommender is a program that suggests items, here songs, that a person is likely to like. This one is content-based: it compares the properties of each song with the properties you say you want. It does not use other people's listening data.

## what is a profile
A profile is the set of settings that describe what you want in a song, such as your target energy, your favorite mood, and the occasion. The app builds it from your answers and you can edit it.

## what is a score
A score is a number from 0 to 1 that says how well a song matches your profile. 1.00 is a perfect match and 0 is no match. Songs with higher scores are shown first.

## what is a weight
A weight says how important one setting is. Genre has a bigger weight than duration, so matching genre earns more points than matching song length. Different ranking modes use different weights.

## what is RAG
RAG stands for retrieval-augmented generation. Before answering your question, the app looks up the most relevant parts of its own reference documents and gives them to the AI, so the AI answers from those documents instead of guessing. This is why explanations of attributes such as energy match the definitions in these docs.

## what is Gemini
Gemini is the large language model from Google that the app uses to talk with you, understand your free-text answers, and write explanations. It never chooses songs by itself. Songs are ranked by the scoring code and come only from the catalog.

## why did you ask me that question
The app asks about the setting it knows least about, starting with the settings that matter most for ranking. Each answer reduces what it does not know. You can press stop at any time.

## why is this song in my top 5
Open the explanation under the song. It lists every setting that was compared, the points the song earned for each, and the total. The songs with the highest totals are shown.

## why did a song with a similar mood not score for mood
Mood is an exact match. "Calm" and "relaxed" are close in meaning, but the scorer treats them as different labels, so the song earns 0 for mood. The other settings can still make the song rank well.

## why did the list change after I rated the songs
Songs you rated high pull your profile toward their properties and songs you rated low push it away. The new list is scored with the updated profile.

## why do I have to rate all 5 songs
Every round should give the app a complete picture of what you liked and disliked. The Regenerate button unlocks only after all 5 songs have a rating.

## can I change my preferences by hand
Yes. The profile tab shows every setting. You can change a value or remove a setting. Removed settings are ignored when scoring.

## what if no song matches
The app still shows the 5 closest songs, with low scores. A low top score means the catalog has no good match for your taste. The app should tell you this instead of pretending.

## is the song data exact
No. The numbers for energy, valence, danceability, and the other attributes were estimated by hand for this project. They are close guides and not measurements.

## can the app recommend a song that is not in the catalog
No. Every recommendation comes from the 50-song catalog. Explanations that mention a song outside the catalog are rejected.

## does the app save my data
Ratings and profile are kept only during your session. The app writes logs of the questions asked and the scores for testing. It does not need your name, and you should not type personal information into the chat.
