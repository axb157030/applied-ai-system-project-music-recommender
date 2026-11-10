# Song attributes: definitions, scales, examples, and how each one is scored

Every song in the catalog has the attributes below. Most are numbers. The recommender compares the number you want (your target) with the song's number, and the closer they are, the more points the song earns for that attribute. Examples are real songs from the catalog with their real catalog values.

## energy
Energy (also called intensity, power, loudness, or how active a song feels) says how intense and active a song is. Low energy is quiet, slow, and gentle. High energy is loud, fast, and driving. It is a number from 0 to 1.
- Low (0.1 to 0.3): Weightless by Marconi Union (0.10), Clair de Lune by Claude Debussy (0.12), Gymnopedie No. 1 by Erik Satie (0.16).
- Medium (0.4 to 0.6): Dreams by Fleetwood Mac (0.50), Here Comes the Sun by The Beatles (0.55), Sunset Lover by Petit Biscuit (0.58).
- High (0.7 to 1.0): Lose Yourself by Eminem (0.73), Blinding Lights by The Weeknd (0.80), Smells Like Teen Spirit by Nirvana (0.91).
Words to numbers: "very calm" or "sleepy" = 0.1, "chill" or "quiet" = 0.25, "moderate" or "easygoing" = 0.5, "upbeat" or "lively" = 0.7, "high energy" or "pumped" = 0.85, "intense" or "extreme" = 0.95.
How it is scored: similarity = 1 minus the gap between your target and the song's energy. A target of 0.20 and a song at 0.12 gives similarity 0.92.

## valence
Valence (also called positivity, happiness, or mood brightness) says how happy and positive a song sounds, from 0 (sad, dark, tense) to 1 (happy, joyful, cheerful). It is not the same as energy: a song can be quiet and happy, or loud and sad.
- Low (0.0 to 0.3): Lose Yourself by Eminem (0.06), Hallelujah by Jeff Buckley (0.10), Mad World by Gary Jules (0.10).
- Medium (0.4 to 0.6): Hotel California by Eagles (0.58), Sunset Lover by Petit Biscuit (0.50).
- High (0.7 to 1.0): Levitating by Dua Lipa (0.91), Shape of You by Ed Sheeran (0.93), September by Earth, Wind & Fire (0.98).
Words to numbers: "sad" or "gloomy" = 0.1, "melancholy" = 0.25, "neutral" = 0.5, "cheerful" = 0.75, "joyful" or "feel-good" = 0.9.
How it is scored: similarity = 1 minus the gap between your target valence and the song's valence.

## tempo
Tempo (also called speed, pace, or BPM, beats per minute) says how fast the beat is. Slow songs are around 60 BPM, walking pace is around 100 BPM, fast dance and workout songs are 120 BPM or more. In the catalog tempo ranges from about 56 to 212 BPM.
- Slow (60 to 80 BPM): Clair de Lune by Claude Debussy (60), Weightless by Marconi Union (60), Hotel California by Eagles (74).
- Medium (90 to 120 BPM): Sunset Lover by Petit Biscuit (91), Levitating by Dua Lipa (103), Uptown Funk by Mark Ronson ft. Bruno Mars (115).
- Fast (130 BPM or more): Gurenge by LiSA (135), Blinding Lights by The Weeknd (171), Master of Puppets by Metallica (212).
Words to numbers: "slow" = 65 BPM, "relaxed pace" = 85, "walking pace" = 100, "upbeat" = 120, "fast" = 150, "very fast" = 180.
How it is scored: similarity = 1 minus the gap divided by 100 BPM, so songs 100 BPM or more away from your target earn 0 for tempo. Note that some catalog tempos are "double time" (a fast-feeling song may be listed at a slow-feeling number), so tempo is a rough guide.

## danceability
Danceability says how easy a song is to dance to, based on steady beat and groove. It is a number from 0 to 1.
- Low (0.1 to 0.3): Weightless by Marconi Union (0.15), Gymnopedie No. 1 by Erik Satie (0.20), Clair de Lune by Claude Debussy (0.30).
- Medium (0.4 to 0.6): Bohemian Rhapsody by Queen (0.41), Here Comes the Sun by The Beatles (0.55).
- High (0.7 to 1.0): Levitating by Dua Lipa (0.70), Shape of You by Ed Sheeran (0.83), Uptown Funk by Mark Ronson ft. Bruno Mars (0.90).
Words to numbers: "not danceable" = 0.15, "some groove" = 0.5, "danceable" = 0.75, "club or party" = 0.9.
How it is scored: similarity = 1 minus the gap between your target and the song's danceability.

## acousticness
Acousticness says how acoustic (real instruments such as piano, guitar, strings, voice) versus electronic (synthesizers, drum machines, studio effects) a song sounds. It is a number from 0 (fully electronic) to 1 (fully acoustic).
- Low (0.0 to 0.2): Blinding Lights by The Weeknd (0.00), UNDERTALE - CORE (RoBKTA Remix) by RoBKTA (0.02), Sunset Lover by Petit Biscuit (0.20).
- Medium (0.3 to 0.6): Weightless by Marconi Union (0.60), Stardew Valley Overture by ConcernedApe (0.50).
- High (0.7 to 1.0): Hallelujah by Jeff Buckley (0.70), Clair de Lune by Claude Debussy (0.98), Moonlight Sonata by Ludwig van Beethoven (0.99).
Words to numbers: "electronic" or "synth" = 0.05, "mixed" = 0.4, "acoustic" or "unplugged" = 0.8, "pure piano or orchestra" = 0.95.
How it is scored: similarity = 1 minus the gap between your target and the song's acousticness.

## instrumentalness
Instrumentalness says how likely a song has no singing. 0 means it has vocals, and values near 1 mean it is instrumental (no lyrics). Use it when someone says "no lyrics", "I cannot focus with words", or "music to study to".
- Low (0.0): Blinding Lights by The Weeknd, Shape of You by Ed Sheeran, Hotel California by Eagles (0.01).
- High (0.85 to 0.98): Take Five by Dave Brubeck Quartet (0.90), Sunset Lover by Petit Biscuit (0.85), Clair de Lune by Claude Debussy (0.95), Gymnopedie No. 1 by Erik Satie (0.98).
Words to numbers: "with vocals" = 0.0, "mostly instrumental" = 0.7, "no lyrics at all" = 0.95.
How it is scored: similarity = 1 minus the gap between your target and the song's instrumentalness. See also the vocals attribute, which is a simple yes or no version of this.

## vocals
Vocals says whether the song has singing. The value is "vocal" (it has a singer) or "instrumental" (no singing). It is an exact match: matching earns full points, a mismatch earns none.
- vocal: Hallelujah by Jeff Buckley, Uptown Funk by Mark Ronson ft. Bruno Mars.
- instrumental: Clair de Lune by Claude Debussy, Cornfield Chase by Hans Zimmer, Megalovania by Toby Fox.

## popularity
Popularity says how widely known a song is, from 0 (hidden gem) to 100 (a global hit). Use it for "mainstream hits" versus "songs I have never heard".
- Lower (60 to 70): Weightless by Marconi Union (60), Undertale - Ruins by Malcolm Robinson Music (66), Deltarune - Beginning by Pennoink (69).
- Higher (90 to 95): Blinding Lights by The Weeknd (95), Shape of You by Ed Sheeran (93), Sweden by C418 (90).
Words to numbers: "hidden gems" or "underground" = 55, "some familiar" = 70, "well known" = 85, "chart hits" = 95.
How it is scored: similarity = 1 minus the gap divided by 100.

## release_decade
Release decade says when the song was first released or composed, written as the first year of the decade (1990 means the 1990s). Classical pieces use their composition decade, for example Gymnopedie No. 1 is 1880 and Clair de Lune is 1900. In the catalog it ranges from 1720 (Spring by Antonio Vivaldi) to 2020.
- 1960s: Here Comes the Sun by The Beatles, At Last by Etta James.
- 1990s: Smells Like Teen Spirit by Nirvana, Hallelujah by Jeff Buckley.
- 2020s: Blinding Lights by The Weeknd, Levitating by Dua Lipa.
How it is scored: similarity = 1 minus the gap in years divided by 50, so songs 50 or more years away from your preferred decade earn 0.

## duration
Duration is the song length in seconds. In the catalog it ranges from about 126 seconds (Cornfield Chase by Hans Zimmer) to 660 seconds (the Dvorak symphony movement).
- Short (under 180 s): Three Little Birds by Bob Marley and The Wailers (180 s), Fly Me to the Moon by Frank Sinatra (148 s), Megalovania by Toby Fox (156 s).
- Medium (180 to 300 s): Blinding Lights (200 s), Clair de Lune (300 s).
- Long (over 360 s): Weightless by Marconi Union (480 s), Master of Puppets by Metallica (515 s), Strobe by deadmau5 (637 s).
How it is scored: similarity = 1 minus the gap divided by 180 seconds, so songs 3 minutes or more away from your target earn 0.

## language
Language is the language of the lyrics: English, Japanese, Portuguese, or "none" for instrumental songs. It is an exact match: matching earns full points, a mismatch earns none.
- English: Hotel California by Eagles.
- Japanese: Gurenge by LiSA, Hotaru no Hikari by Rainych Ran.
- Portuguese: Mas Que Nada by Sergio Mendes and Brasil '66.
- none: Clair de Lune by Claude Debussy.

## explicit
Explicit means the song contains strong language or adult themes, the same label streaming services show with an "E". In the catalog the value is "yes" or "no". Explicit songs in the catalog include Lose Yourself by Eminem and SICKO MODE by Travis Scott.
How it works: it is a filter, not a score. If the user says no explicit songs, every explicit song is removed before ranking, no matter how well it matches. If the user does not say, nothing is filtered.

## speechiness
Speechiness says how much of the song is spoken words rather than singing, for example rap, spoken poetry, or talking. It is a number from 0 to 1. Rap songs such as Lose Yourself by Eminem (0.30) are high in this catalog; most songs are near 0.05.
Note: speechiness is stored for every song, but the current recommender does not score it. It is available for future use.
