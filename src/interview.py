"""
Interviewer: the conversation users actually talk to.

WHO DOES WHAT
    retriever.py    finds the doc paragraphs about a topic ("what is energy?").
    llm.py          Gemini: writes interview questions, answers from retrieved
                    paragraphs (RAG), and turns free text into preferences
                    (extract_preferences).
    preferences.py  checks everything Gemini returns before it is trusted.
    recommender.py  scores the song catalog against the preferences.
    interview.py    (this file) decides which of the above to call on each turn.

EACH TURN
    1. Show a prompt: the opening question, an interview question about the
       attribute we know least about, or a free prompt once we have enough.
    2. Read what the user typed and decide what it is:
         command    quit, recommend, rate, profile, skip, refine, reset
         question   ends with "?"  -> answer from the docs (RAG), then re-ask
         otherwise  a preference   -> extract_preferences, merge into the profile
                    (if nothing can be extracted, try it as a question, since
                    people often leave off the "?": "what is tempo")
    3. Once the profile has enough preferences, run recommend_songs and show the
       top songs with the reasons they scored well. Further preferences refine them.
    4. "rate": the user gives each pick 1-10 (and an optional comment). feedback.py
       measures the round, nudges the profile, and a new list of unseen songs is shown.

RAG answers the user's questions. It never supplies the user's preferences:
those come only from what the user says.

Every decision is written to logs/interview_<time>.txt (the "reasoning trace").

The console (python -m src.main) and the Streamlit app (app.py) both drive this same
class: they call respond(text) for each message and prompt() for the next question.
"""

import os
from datetime import datetime

from .llm import GeminiClient, LLMError
from .preferences import (
    MIN_PREFERENCES,
    build_vocab,
    clean_preferences,
    enough_to_recommend,
    unknown_attributes,
)
from .recommender import load_songs, recommend_songs
from .retriever import Retriever
from .feedback import LIKE_AT_LEAST, MAX_RATING, MIN_RATING, regenerate
from .guardrails import check_input

TOP_K = 5                 # songs shown per recommendation
MAX_QUESTIONS = 6         # interview questions before we stop asking
MAX_CLARIFICATIONS = 3    # side questions allowed while one question is pending
REFINE_EXTRA = 3          # extra questions when the user types "refine"

QUIT_WORDS = {"quit", "exit", "bye", "goodbye", "q"}
RECOMMEND_WORDS = {"recommend", "recommendations", "done", "songs", "show songs"}
RATE_WORDS = {"rate", "rate songs", "rate these", "ratings"}
PROFILE_WORDS = {"profile", "my profile", "preferences"}
RESET_WORDS = {"reset", "start over"}
REFINE_WORDS = {"refine", "more", "keep going", "ask me more"}
SKIP_WORDS = {
    "skip", "pass", "no preference", "no idea", "not sure", "idk",
    "dont care", "don't care", "any", "anything", "whatever",
}

# The docs write a line "Words to numbers: ..." under each attribute. extraction_reference()
# collects those lines for ALL attributes, so Gemini can always turn "acoustic" or "fast"
# into a number, whichever paragraphs the retriever happened to rank first.
WORDS_TO_NUMBERS = "Words to numbers:"

GREETING = (
    "Hi! I'll help you find songs you'll like.\n"
    "Tell me what you're in the mood for, or ask me anything, like "
    "\"what is energy?\" or \"what is tempo?\".\n"
    "Commands: recommend, rate, profile, skip, refine, reset, quit."
)
OPENING_QUESTION = (
    "What kind of music are you looking for? "
    "(a mood, an occasion, a genre... anything you like)"
)
FREE_PROMPT = (
    "Tell me more about what you like, ask me a question, "
    "or type 'refine', 'recommend', 'rate' or 'quit'."
)
NOT_IN_DOCS = (
    "I couldn't find that in my reference docs. I can explain things like "
    "energy, tempo, valence, acousticness, moods and genres."
)
GEMINI_DOWN = "I couldn't reach Gemini just now. Please try again in a moment."
NO_TASTE_YET = (
    "I don't know your taste yet. Tell me about a mood, an occasion, or a genre you like."
)

# Used when Gemini cannot write the question, so the interview still works offline.
FALLBACK_QUESTIONS = {
    "favorite_mood": "What mood are you looking for in your music?",
    "preferred_occasions": "What will you listen to it for (studying, working out, relaxing...)?",
    "target_energy": "Do you want calm, medium, or high-energy music?",
    "favorite_genre": "Is there a genre you prefer?",
    "vocals": "Do you prefer songs with vocals, or instrumentals?",
    "target_valence": "Should the music sound sad, neutral, or happy?",
    "target_acousticness": "Do you prefer acoustic or electronic sounds?",
    "language": "Is there a language you want the lyrics in?",
    "preferred_decade": "Is there a decade whose music you like most?",
    "target_popularity": "Do you want well-known hits or hidden gems?",
    "target_tempo": "Do you want slow or fast music?",
    "target_danceability": "Should it be easy to dance to?",
    "target_duration": "Do you prefer short songs or longer ones?",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _normalize(text):
    """Lowercase, straighten apostrophes, drop trailing . or ! (for command matching)."""
    return text.lower().replace("\u2019", "'").strip().rstrip(".!").strip()


def _is_refusal(answer):
    """True for either refusal sentence (retriever.py and llm.py word it differently)."""
    return answer.strip().lower().startswith("i do not know")


def _attribute_query(attr):
    """"target_energy" -> "energy", "preferred_occasions" -> "occasions"."""
    for prefix in ("target_", "favorite_", "preferred_"):
        if attr.startswith(prefix):
            attr = attr[len(prefix):]
    return attr.replace("_", " ")


def _format_snippets(snippets):
    if not snippets:
        return "(no reference found)"
    return "\n\n".join(f"[{name}]\n{text}" for name, text in snippets)


def _describe(prefs):
    """{"favorite_mood": "chill", "preferred_occasions": ["study"]} -> readable text."""
    parts = []
    for key, value in prefs.items():
        if isinstance(value, list):
            value = ", ".join(value)
        parts.append(f"{key.replace('_', ' ')}: {value}")
    return "; ".join(parts)


def _catalog_snippet(vocab):
    """
    A pseudo-document listing the labels that really exist in the song catalog.
    It is added to the reference given to extract_preferences so Gemini picks
    labels the recommender can match instead of inventing new ones.
    """
    lines = ["Labels that exist in the song catalog (use only these):"]
    for key, name in (
        ("favorite_genre", "genres"),
        ("favorite_mood", "moods"),
        ("preferred_occasions", "occasions"),
        ("preferred_tags", "mood tags"),
    ):
        values = sorted(vocab.get(key, ()))
        if values:
            lines.append(f"{name}: {', '.join(values)}")
    lines.append('vocals: "vocal" or "instrumental"')
    return ("song catalog", "\n".join(lines))


def _words_to_numbers_snippet(retriever):
    """One pseudo-document with the "Words to numbers" line of every attribute in the docs."""
    lines = []
    for _name, text in getattr(retriever, "chunks", []):
        paragraph = text.splitlines()
        heading = paragraph[0].lstrip("# ").strip()
        lines += [f"{heading}: {line[len(WORDS_TO_NUMBERS):].strip()}"
                  for line in paragraph if line.startswith(WORDS_TO_NUMBERS)]
    return ("word-to-number tables", "\n".join(lines)) if lines else None


def extraction_reference(retriever, vocab, queries):
    """
    Reference text for extract_preferences:
      - the doc paragraphs the retriever finds for each query
      - the word-to-number tables of every attribute
      - the labels that really exist in the song catalog
    """
    snippets = []
    for query in queries:
        for item in retriever.retrieve(query):
            if item not in snippets:
                snippets.append(item)
    tables = _words_to_numbers_snippet(retriever)
    if tables:
        snippets.append(tables)
    snippets.append(_catalog_snippet(vocab))
    return snippets


def _reasons_text(explanation):
    """The recommender's explanation without its final 'total: ...' line."""
    parts = [p for p in str(explanation).split("; ") if p and not p.startswith("total:")]
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The interviewer
# ---------------------------------------------------------------------------

class Interview:
    """
    One conversation with one listener.

    songs:     list of song dicts from load_songs()
    retriever: Retriever (built with llm_client=llm so answer_rag works)
    llm:       GeminiClient (anything with ask() and extract_preferences())
    ask_user / say: injectable input and output, so tests can script a conversation.
    """

    def __init__(self, songs, retriever, llm=None, profile=None, ask_user=input, say=print,
                 log_dir="logs", k=TOP_K, max_questions=MAX_QUESTIONS):
        self.songs = songs
        self.retriever = retriever
        # Created here, not as a default argument: a default is built when the file is
        # imported, which would crash any import without a GEMINI_API_KEY.
        self.llm = llm if llm is not None else GeminiClient()
        self.vocab = build_vocab(songs)
        # Starting profile must use UserProfile key names (favorite_mood, ...).
        # clean_preferences silently drops anything else.
        self.profile = clean_preferences(profile or {}, self.vocab)
        self.ask_user = ask_user
        self.say = say
        self.log_dir = log_dir
        self.k = k
        self.max_questions = max_questions

        self.skipped = set()          # attributes the user could not or would not answer
        self.pending_attr = None      # attribute currently being asked about
        self.pending_question = None  # the text of that question (re-asked after side questions)
        self.clarifications = 0       # side questions asked while pending_attr is open
        self.questions_asked = 0
        self.opened = False           # has the user answered the opening question yet?
        self.gave_up = False          # stopped asking before we had enough preferences
        self.refining = False         # user typed "refine": keep asking even though ready
        self.last_recommendations = []
        self.rated_ids = set()        # songs the user has rated; never shown again
        self.history = []             # metrics of each rated round (see feedback.compute_metrics)
        self._log = None

    # -----------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------

    def open_log(self):
        """Start the reasoning-trace file. run() calls this; the Streamlit app calls it once."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"interview_{datetime.now():%Y%m%d_%H%M%S}.txt")
        self._log = open(path, "w", encoding="utf8")
        self._trace(f"Initial profile: {self.profile}")

    def close_log(self):
        if self._log:
            self._log.close()
            self._log = None

    def run(self):
        """Console conversation: talk to the user until they quit. Returns the last picks."""
        self.open_log()
        try:
            self._converse()
        finally:
            self.close_log()
        return self.last_recommendations

    def prompt(self):
        """The next thing to ask the user (an interview question or a free prompt)."""
        return self._next_prompt()

    def respond(self, text):
        """
        Handle one message from the user. Returns "quit" when they want to leave, else None.
        Everything to show the user goes through self.say.
        """
        text = text.strip()
        if not text:
            return None
        verdict = check_input(text)
        if not verdict.allowed:
            # Log which rule fired, never the text itself (it may hold personal data).
            self._trace(f"Blocked input (category: {verdict.category})")
            self.say(verdict.message)
            return None
        self._trace(f"User: {text!r}")
        self.opened = True
        return self._handle(text)

    def _converse(self):
        self.say(GREETING)
        while True:
            prompt = self._next_prompt()
            prompt = f"{prompt}\n> " if prompt else "> "
            try:
                text = self.ask_user(prompt)
            except (EOFError, KeyboardInterrupt):
                self._trace("Input closed; ending.")
                break
            if self.respond(text) == "quit":
                break
        self.say("Goodbye! Enjoy the music.")

    def _trace(self, message):
        if self._log:
            self._log.write(message + "\n")
            self._log.flush()  # so the trace survives a crash

    # -----------------------------------------------------------
    # Readiness and choosing what to ask
    # -----------------------------------------------------------

    def _ready(self):
        """Enough preferences to recommend. After we stop asking, even one is accepted."""
        minimum = 1 if self.gave_up else MIN_PREFERENCES
        return enough_to_recommend(self.profile, minimum=minimum)

    def _next_prompt(self):
        if not self.opened:
            return OPENING_QUESTION
        self._choose_pending()
        return self.pending_question if self.pending_attr else FREE_PROMPT

    def _choose_pending(self):
        """Pick the next attribute to ask about, unless one is open or we are done asking."""
        if self.pending_attr is not None or self.gave_up:
            return
        if self._ready() and not self.refining:
            return

        candidates = [a for a in unknown_attributes(self.profile) if a not in self.skipped]
        if self.questions_asked >= self.max_questions or not candidates:
            self._stop_asking()
            return

        attr = candidates[0]
        self.pending_attr = attr
        self.clarifications = 0
        self.questions_asked += 1
        self._trace(f"Unknown attributes: {candidates}; asking about {attr!r}")
        self.pending_question = self._make_question(attr)

    def _stop_asking(self):
        was_refining = self.refining
        self.refining = False
        if self._ready():
            if was_refining:
                self.say("I don't have any more questions. Ask me anything, or type 'quit'.")
            return
        self.gave_up = True
        self._trace("Stopped asking before reaching the minimum; accepting any preference.")
        if self._ready():
            self.say("I'll work with what I have.")
            self._recommend()
        else:
            self.say(NO_TASTE_YET)

    def _make_question(self, attr):
        """Gemini writes one question using the docs and the catalog's real labels."""
        snippets = self.retriever.retrieve(_attribute_query(attr))
        self._trace(f"Retrieved for {attr!r}: {[name for name, _ in snippets]}")

        options = sorted(self.vocab.get(attr, ()))
        options_line = f"Valid options in the catalog: {', '.join(options)}\n\n" if options else ""
        prompt = (
            f"You are interviewing a music listener. Write ONE short question asking "
            f"about their preference for '{attr}'. If valid options are listed, offer "
            f"only those. Use the reference to explain terms, but never mention scoring, "
            f"weights, the scorer or how the app works inside. Reply with just the "
            f"question.\n\n{options_line}Reference:\n{_format_snippets(snippets)}"
        )
        try:
            question = self.llm.ask(prompt).strip()
        except LLMError as error:
            self._trace(f"Gemini failed writing the question: {error}")
            question = ""
        if not question:
            question = FALLBACK_QUESTIONS.get(attr, f"What do you prefer for {attr.replace('_', ' ')}?")
        self._trace(f"Question: {question}")
        return question

    # -----------------------------------------------------------
    # Deciding what the user typed
    # -----------------------------------------------------------

    def _handle(self, text):
        command = _normalize(text)
        if command in QUIT_WORDS:
            return "quit"
        if command in RESET_WORDS:
            self._reset()
        elif command in PROFILE_WORDS:
            self._show_profile()
        elif command in RECOMMEND_WORDS:
            self._recommend()
        elif command in RATE_WORDS:
            self._rate()
        elif command in REFINE_WORDS:
            self._start_refining()
        elif command in SKIP_WORDS:
            self._skip_pending("user chose to skip")
        elif text.endswith("?"):
            self._answer_question(text)
        else:
            self._take_preferences(text)
        return None

    # -----------------------------------------------------------
    # Questions (RAG)
    # -----------------------------------------------------------

    def _ask_docs(self, text):
        """Returns ("answer", text), ("refused", "") or ("error", "")."""
        try:
            answer = self.retriever.answer_rag(text)
        except LLMError as error:
            self._trace(f"RAG failed: {error}")
            return "error", ""
        if _is_refusal(answer):
            self._trace(f"Docs have no answer for {text!r}")
            return "refused", ""
        self._trace(f"RAG answer: {answer}")
        return "answer", answer

    def _answer_question(self, text):
        self._trace(f"Treating as a question: {text!r}")
        status, answer = self._ask_docs(text)
        self.say({"answer": answer, "refused": NOT_IN_DOCS, "error": GEMINI_DOWN}[status])
        self._count_clarification()

    def _count_clarification(self):
        """A pending question gets MAX_CLARIFICATIONS side questions, then we move on."""
        if self.pending_attr is None:
            return
        self.clarifications += 1
        if self.clarifications >= MAX_CLARIFICATIONS:
            self._skip_pending("too many side questions", "Let's move on to something else.")

    # -----------------------------------------------------------
    # Preferences (extract_preferences)
    # -----------------------------------------------------------

    def _reference_for(self, text):
        """Docs for extract_preferences: about what the user said, about the pending
        attribute, the words-to-numbers tables, and the catalog's real labels."""
        queries = [text]
        if self.pending_attr:
            queries.append(_attribute_query(self.pending_attr))
        snippets = extraction_reference(self.retriever, self.vocab, queries)
        self._trace(f"Reference for extraction: {[name for name, _ in snippets]}")
        return snippets

    def _take_preferences(self, text):
        try:
            extracted = self.llm.extract_preferences(text, self._reference_for(text), self.vocab)
        except LLMError as error:
            self._trace(f"Gemini failed extracting preferences: {error}")
            self.say(GEMINI_DOWN)
            return

        # llm.py already validates, but this is cheap and keeps the profile safe
        # even if extract_preferences changes.
        extracted = clean_preferences(extracted, self.vocab)
        self._trace(f"Extracted: {extracted}")

        if not extracted:
            self._no_preference_found(text)
            return

        asked = self.pending_attr
        self.profile.update(extracted)
        self.say("Got it: " + _describe(extracted) + ".")
        if asked:
            if asked not in extracted:
                self.skipped.add(asked)
                self._trace(f"Reply did not answer {asked!r}; skipping it.")
            self._clear_pending()
        self._trace(f"Profile now: {self.profile}")
        if self._ready():
            self._recommend()

    def _no_preference_found(self, text):
        """Nothing to extract. It may be a question typed without "?" ("what is tempo")."""
        status, answer = self._ask_docs(text)
        if status == "answer":
            self.say(answer)
            self._count_clarification()
        elif status == "error":
            self.say(GEMINI_DOWN)
        elif self.pending_attr:
            self._skip_pending("no usable preference in the reply",
                               "I didn't catch a preference there, so let's move on.")
        else:
            self.say("I didn't catch a preference in that. "
                     "Try something like 'calm songs for studying'.")

    # -----------------------------------------------------------
    # Commands
    # -----------------------------------------------------------

    def _clear_pending(self):
        self.pending_attr = None
        self.pending_question = None
        self.clarifications = 0

    def _skip_pending(self, reason, message="Okay, we'll skip that one."):
        if self.pending_attr is None:
            self.say("There's nothing to skip right now.")
            return
        self.skipped.add(self.pending_attr)
        self._trace(f"Skipping {self.pending_attr!r}: {reason}")
        self._clear_pending()
        self.say(message)

    def _start_refining(self):
        if not enough_to_recommend(self.profile, minimum=1):
            self.say(NO_TASTE_YET)
            return
        self.refining = True
        self.gave_up = False
        self.max_questions += REFINE_EXTRA
        self.say("Okay, a few more questions to sharpen the picks.")

    def _show_profile(self):
        if not self.profile:
            self.say("I don't know your preferences yet.")
        else:
            self.say("What I know so far: " + _describe(self.profile) + ".")

    def _reset(self):
        self.profile = {}
        self.skipped.clear()
        self._clear_pending()
        self.questions_asked = 0
        self.gave_up = False
        self.refining = False
        self.opened = False
        self.last_recommendations = []
        self.rated_ids = set()
        self.history = []
        self._trace("Reset.")
        self.say("Okay, starting over.")

    # -----------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------

    def _recommend(self):
        if not enough_to_recommend(self.profile, minimum=1):
            self.say(NO_TASTE_YET)
            return
        pool = [song for song in self.songs if song["id"] not in self.rated_ids]
        self._show_picks(recommend_songs(self.profile, pool, k=self.k))

    def _show_picks(self, results):
        self.last_recommendations = results
        self._trace("Recommendations: " + "; ".join(
            f"{song.get('title')} ({score:.2f})" for song, score, _ in results))

        if not results:
            self.say("No songs in the catalog match those preferences.")
            return

        self.say("Here are your top picks:")
        for rank, (song, score, explanation) in enumerate(results, start=1):
            title = song.get("title", "(untitled)")
            artist = song.get("artist", "unknown artist")
            self.say(f"{rank}. {title} by {artist} (match {score:.0%})")
            why = _reasons_text(explanation)
            if why:
                self.say(f"   why: {why}")
        self.say("Type 'rate' to rate these 1-10 and get a better list, "
                 "tell me more to refine them, or type 'quit'.")

    # -----------------------------------------------------------
    # Ratings (feedback.py)
    # -----------------------------------------------------------

    def _rate(self):
        """Console flow: ask for a 1-10 rating (and an optional comment) for each pick."""
        if not self.last_recommendations:
            self.say("There are no songs to rate yet. Type 'recommend' first.")
            return
        self.say(f"Rate each song {MIN_RATING}-{MAX_RATING}. You can add a comment after the "
                 f"number, like '4 too slow'. Type 'cancel' to stop.")
        ratings, comments = {}, {}
        for song, _score, _why in self.last_recommendations:
            while True:
                answer = self.ask_user(f"  {song['title']} by {song['artist']}: ").strip()
                if answer.lower() == "cancel":
                    self.say("Okay, no ratings saved.")
                    return
                number, _, comment = answer.partition(" ")
                if number.isdigit() and MIN_RATING <= int(number) <= MAX_RATING:
                    ratings[song["id"]] = int(number)
                    if comment.strip():
                        comments[song["id"]] = comment.strip()
                    break
                self.say(f"Please start with a number from {MIN_RATING} to {MAX_RATING}.")
        self.submit_ratings(ratings, comments)

    def submit_ratings(self, ratings, comments=None):
        """
        Use the ratings for the songs last shown: measure the round, improve the profile
        and show a new list. Raises ValueError for invalid ratings. The console and the
        Streamlit app both call this.
        """
        shown = [song["id"] for song, _score, _why in self.last_recommendations]
        snippets = None
        if comments:
            snippets = extraction_reference(self.retriever, self.vocab, [])
        result = regenerate(
            self.profile, ratings, self.songs, shown_ids=shown, k=self.k,
            exclude_ids=self.rated_ids, comments=comments, llm=self.llm,
            snippets=snippets, vocab=self.vocab)

        metrics = result["metrics"]
        self.rated_ids |= set(ratings)
        self.history.append(metrics)
        self._trace(f"Ratings: {ratings}; metrics: {metrics}")
        self._trace(f"Profile before: {self.profile}")
        self.profile = result["profile"]
        self._trace(f"Profile after:  {self.profile}")

        self.say(f"{result['headline']} (average {metrics['mean']}/10, "
                 f"{metrics['hit_rate']:.0%} rated {LIKE_AT_LEAST}+)")
        for note in result["notes"]:
            self._trace(note)
            self.say(note)
        if not any(key.startswith("target_") for key in self.profile):
            self.say("Your profile has no numeric targets (like energy) yet, so the ratings only "
                     "removed those songs from the list. Tell me about energy or tempo to go further.")
        self._show_picks(result["recommendations"])
        return result


def main(csv_path="data/songs.csv", docs_folder="docs"):
    try:
        llm = GeminiClient()
    except RuntimeError as error:   # missing GEMINI_API_KEY
        print(error)
        return
    songs = load_songs(csv_path)
    retriever = Retriever(docs_folder, llm_client=llm)
    Interview(songs, retriever, llm).run()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])