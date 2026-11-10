# How this music recommender works

## how this application works (overview)
This application recommends songs from a catalog of 50 real songs. It does not invent songs and it does not use other people's listening history. You describe what you like (in your own words or by moving sliders), the app turns that into a profile of numbers and labels, scores every song against the profile, and shows your top 5 with the reasons. You then rate the 5 songs from 1 to 10, and the app uses your ratings to improve the next list.

## the interview
Many people do not know their exact taste, so the app asks questions instead of asking for numbers. It asks about one topic at a time: mood and occasion, energy and pace, sound texture (acoustic or electronic, vocals or not), genres and artists you know, era and how famous the songs are, and practical limits (song length, language, explicit songs). One answer can fill several settings at once. For example "something to study to, no lyrics" sets the occasion to study, vocals to instrumental, and a low energy. The interview stops when it is confident about the important settings, when it has asked the maximum number of questions, or when you press stop.

## the profile
The profile is the list of things you want: favorite genre, favorite mood, target energy, target valence, target tempo, target danceability, target acousticness, target instrumentalness, vocals or instrumental, language, wanted occasions, wanted mood tags, preferred decade, target popularity, target duration, and whether explicit songs are allowed. Every setting is optional. Only the settings you give are used to score songs. You can see and edit every setting in the profile tab.

## scoring
Each song gets a score from 0 to 1. For every setting in your profile, the song earns similarity points between 0 and 1: exact-match settings (mood, language, vocals) earn 1 or 0, and number settings earn 1 minus how far the song is from your target. Each setting has a weight that says how important it is. The final score is the weighted points earned divided by the weighted points possible, so a perfect match is 1.00. Each song's explanation lists every setting, the points earned, and the total, so you can see exactly why it ranked where it did.

## ranking modes
There are four ranking modes (strategies). Each uses different weights:
- balanced (default): genre 0.35 and mood 0.30 matter most; energy, acousticness, tags, and the rest add detail.
- genre_first: genre decides almost everything (0.60).
- mood_first: mood (0.55) and mood tags (0.25) decide most of the ranking.
- energy_focused: closeness of energy (0.55) decides most of the ranking.

## variety rule
To stop one artist from filling the whole list, each song loses 0.10 points for every song by the same artist that was already picked.

## explicit filter
If you say you do not want explicit songs, all explicit songs are removed before ranking.

## explanations
Gemini writes a short plain-language explanation for each recommended song, based on the song's real catalog values and the reference docs. Every number it mentions is checked against the catalog. If the explanation contains a wrong number or a song that is not in the catalog, it is rejected and replaced by the plain score breakdown.

## ratings and feedback
After you rate all 5 songs from 1 (dislike) to 10 (love), the app shows the average rating and how many songs you rated 7 or higher. It then moves your profile toward the songs you rated high and away from the songs you rated low. Pressing Regenerate gives a new top 5 from the updated profile. You must rate all 5 first so every round gives the system real feedback. You can add an optional comment on any song, such as "too slow", which can be used to adjust the profile.

## listening
Each song has a Listen link that opens a YouTube search for that song. The app does not play or store audio itself. Whether a song is available, and whether there are ads, depends on YouTube.

## limits
The catalog has only 50 songs, so some tastes have no good match. Song attribute values were estimated by hand and are not precise. Mood must match exactly, so "calm" and "relaxed" count as different moods in scoring. Ratings from one user in a few rounds are a small, noisy sample. The recommender is content-based: it never learns from other users.
