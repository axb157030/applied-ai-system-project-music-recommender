from .recommender import load_songs, recommend_songs

songs = load_songs("data/songs.csv")
profile = {
    "genre": "classical", "mood": "calm",
    "target_energy": 0.2, "target_acousticness": 0.9,
    "preferred_occasions": ["study"],
}

recommendations = recommend_songs(profile, songs, k=5)

for rank, (song, score, reasons) in enumerate(recommendations, start=1):
    print(f"{rank}. {song['title']} - {song['artist']}  (score {score:.2f})")
    print(f"   {reasons}\n")