"""
Streamlit web UI for the music recommender.

    streamlit run app.py

HOW THIS FILE WORKS
    All of the thinking lives in src/interview.py (the same class the console uses).
    This file only does three things:
      1. Chat: send what the user types to Interview.respond() and show what comes back.
      2. Ratings: show a 1-10 control for each recommended song. The Regenerate button
         stays disabled until every song has a rating.
      3. Sidebar: show what the app has learned about the user and how ratings changed.

HOW STREAMLIT WORKS (the one idea you need)
    Streamlit re-runs this whole file from top to bottom every time the user clicks or
    types. Normal variables are reset on every run, so anything that must survive
    (the Interview object, the chat history) is stored in st.session_state.
"""

from pathlib import Path

import streamlit as st

from src.feedback import MAX_RATING, MIN_RATING
from src.interview import GREETING, Interview
from src.llm import GeminiClient
from src.recommender import load_songs
from src.retriever import Retriever

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")


# ---------------------------------------------------------------------------
# Setup (runs once per browser session)
# ---------------------------------------------------------------------------

def new_session():
    """Create the Interview and the chat history. Called on first load and on 'Start over'."""
    outbox = []                                  # lines the Interview wants to show the user
    llm = GeminiClient()                         # raises RuntimeError when GEMINI_API_KEY is missing
    here = Path(__file__).resolve().parent         # so it works wherever streamlit is launched from
    songs = load_songs(str(here / "data" / "songs.csv"))
    retriever = Retriever(str(here / "docs"), llm_client=llm)

    interview = Interview(songs, retriever, llm, say=outbox.append)
    interview.open_log()                         # writes the reasoning trace to logs/

    st.session_state.interview = interview
    st.session_state.outbox = outbox
    st.session_state.chat = [{"role": "assistant", "text": GREETING},
                             {"role": "assistant", "text": interview.prompt()}]


def collect_replies(interview, outbox):
    """
    Turn everything the Interview said since the last call into one assistant message,
    then add its next question. (The Interview "says" things by calling outbox.append.)
    """
    lines = list(outbox)
    outbox.clear()
    question = interview.prompt()                # may add more lines to the outbox
    lines += outbox
    outbox.clear()
    if question:
        lines.append(question)
    return {"role": "assistant", "text": "\n\n".join(lines)}


if "interview" not in st.session_state:
    try:
        new_session()
    except RuntimeError as error:                # missing API key: show it instead of crashing
        st.error(str(error))
        st.stop()

interview = st.session_state.interview
outbox = st.session_state.outbox
chat = st.session_state.chat


# ---------------------------------------------------------------------------
# Sidebar: what the app knows, and how the ratings are going
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("What I know about you")
    if interview.profile:
        st.json(interview.profile)
    else:
        st.caption("Nothing yet. Tell me what you like.")

    if interview.history:
        st.header("Your ratings")
        st.caption("Average rating of each round. If the app is learning, the line goes up.")
        st.line_chart([round_["mean"] for round_ in interview.history])
        last = interview.history[-1]
        st.metric("Last round: average", f"{last['mean']}/10")
        st.metric("Last round: songs you liked (7+)", f"{last['hit_rate']:.0%}")

    st.divider()
    st.caption("Try: 'what is energy?', 'calm songs for studying', 'profile', 'refine', 'skip'.")
    if st.button("Start over"):
        interview.close_log()
        for key in ("interview", "outbox", "chat"):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: the chat
# ---------------------------------------------------------------------------

st.title("🎵 Music Recommender")
st.caption("Tell me what you like, or ask what a term means. I'll find songs and learn from your ratings.")

for message in chat:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

text = st.chat_input("Describe the music you want, or ask a question")
if text:
    chat.append({"role": "user", "text": text})
    with st.spinner("Thinking..."):
        if interview.respond(text) == "quit":    # preferences, questions, commands, guardrails
            outbox.append("Goodbye! Enjoy the music. (Type anything to keep going.)")
        chat.append(collect_replies(interview, outbox))
    st.rerun()


# ---------------------------------------------------------------------------
# Ratings: shown whenever there are recommendations
# ---------------------------------------------------------------------------

picks = interview.last_recommendations
if picks:
    st.divider()
    st.subheader("Rate these songs")
    st.caption(f"Give each song {MIN_RATING} (hated it) to {MAX_RATING} (loved it). "
               "A short comment like 'too slow' is optional and also teaches the app.")

    round_number = len(interview.history)        # changes after each regeneration, so the widgets reset
    ratings, comments = {}, {}
    for song, score, _why in picks:
        left, middle, right = st.columns([3, 5, 3])
        with left:
            link = f"  \n[listen]({song['url']})" if song.get("url") else ""
            st.markdown(f"**{song['title']}**  \n{song['artist']}{link}")
        with middle:
            # index=None means "nothing chosen yet", which is how we know the user hasn't rated it.
            rating = st.radio("Rating", list(range(MIN_RATING, MAX_RATING + 1)), index=None,
                              horizontal=True, key=f"rating-{round_number}-{song['id']}",
                              label_visibility="collapsed")
        with right:
            comment = st.text_input("Comment", key=f"comment-{round_number}-{song['id']}",
                                    placeholder="optional, e.g. too slow",
                                    label_visibility="collapsed")
        if rating is not None:
            ratings[song["id"]] = rating
        if comment.strip():
            comments[song["id"]] = comment.strip()

    all_rated = len(ratings) == len(picks)
    if st.button("Regenerate", type="primary", disabled=not all_rated,
                 help=None if all_rated else "Rate every song to unlock this button."):
        try:
            with st.spinner("Learning from your ratings..."):
                interview.submit_ratings(ratings, comments)
                chat.append(collect_replies(interview, outbox))
        except ValueError as error:              # should not happen (the button is gated), but be safe
            st.error(str(error))
        else:
            st.rerun()
