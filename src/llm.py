"""
Gemini client wrapper.

Reads the API key from GEMINI_API_KEY (in .env.local or the environment) and offers:
- ask:                  one prompt in, text out (retries once, raises LLMError on failure)
- answer_without_docs:  baseline, the model answers from general knowledge (for comparison)
- answer_from_snippets: RAG, the model may only use the snippets it is given
- extract_preferences:  turns what the user said into a validated preferences dict

Every method goes through ask(), so retry and error handling live in one place.
Failures raise LLMError. They are never returned as text, so callers cannot mistake an
error message for an answer.
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .preferences import parse_preferences_json

load_dotenv(".env.local")   # reads GEMINI_API_KEY from .env.local
load_dotenv()               # also accepts a plain .env

# Model used for every request. Change it here to switch models.
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"

REFUSAL = "I do not know based on the docs I have."


class LLMError(Exception):
    """Raised when a Gemini request fails (network error, quota, invalid key, ...)."""


class GeminiClient:
    """
    Thin wrapper around the Gemini model.

    Usage:
        client = GeminiClient()
        client.answer_from_snippets("what is energy?", snippets)   # RAG
        client.extract_preferences("calm songs for studying", snippets)
    """

    def __init__(self):
        # Fail early with a clear message if the key is missing.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in .env.local or your shell to enable LLM features."
            )
        self.client = genai.Client(api_key=api_key)

    # -----------------------------------------------------------
    # Core call: everything else uses this
    # -----------------------------------------------------------

    def ask(self, prompt, as_json=False, temperature=None):
        """Send one prompt to Gemini and return the reply text. Tries twice, then raises LLMError."""
        config_args = {}
        if as_json:
            config_args["response_mime_type"] = "application/json"
        if temperature is not None:
            config_args["temperature"] = temperature
        config = types.GenerateContentConfig(**config_args) if config_args else None

        last_error = None
        for _attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL_NAME, contents=prompt, config=config)
                return (response.text or "").strip()
            except Exception as e:
                last_error = e
        raise LLMError(f"Gemini call failed: {last_error}")

    # -----------------------------------------------------------
    # Baseline: generation with no retrieval
    # -----------------------------------------------------------

    def answer_without_docs(self, query):
        """Answer from the model's general knowledge only. No docs are sent (the no-RAG baseline)."""
        prompt = f"""You are a helpful assistant for a music recommendation app.
Answer this user question: {query}"""
        return self.ask(prompt)

    # -----------------------------------------------------------
    # RAG: generation grounded in retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples chosen by the retriever.
        Returns the fixed refusal sentence when there is nothing to answer from.
        """
        # Nothing was retrieved: refuse without spending an API call.
        if not snippets:
            return REFUSAL

        # Label each snippet with its filename so the model can cite its sources.
        context = "\n\n".join(f"File: {filename}\n{text}\n" for filename, text in snippets)

        prompt = f"""You are the help assistant of a music recommendation app. You answer
questions from users about music attributes and about how the app works.

Snippets from the app's reference documents:
{context}

User question:
{query}

Rules:
- Use only the information in the snippets. Do not invent songs, attributes, numbers
  or app features that are not in the snippets.
- If the snippets are not enough to answer confidently, reply exactly:
  "{REFUSAL}"
- Keep the answer short (2 to 4 sentences), in plain language, and give an example
  song from the snippets when one is available.
- End by naming the files you relied on."""
        return self.ask(prompt)

    # -----------------------------------------------------------
    # Interview helper: free text -> structured preferences
    # -----------------------------------------------------------

    def extract_preferences(self, user_text, snippets, vocab=None):
        """
        Turn the user's free-text description into a dict of UserProfile fields.

        user_text: e.g. "something chill and acoustic for studying, no explicit songs"
        snippets:  [(filename, text), ...] from the retriever. They must include the genre,
                   mood and occasion lists and the "words to numbers" tables, because the
                   model is told to use only those.

        vocab:     optional, from preferences.build_vocab(songs). Labels not in the catalog are dropped.

        Returns a validated dict that may be missing keys ("the user did not say").
        Returns {} if the reply is not usable JSON. Raises LLMError if Gemini is unreachable.
        """
        reference = "\n\n".join(f"[{name}]\n{text}" for name, text in snippets)

        prompt = f"""You convert a listener's request into music preferences.

Use ONLY the reference below to choose genre, mood and occasion labels, and to turn
words like "high energy" into numbers. Use the exact labels from the reference. If the
user's wording has no matching label, use null.

Return one JSON object with these keys. Use null for anything the user did not say.
Do not guess.
  favorite_genre: string
  favorite_mood: string
  target_energy: number 0-1
  target_acousticness: number 0-1
  target_popularity: number 0-100
  preferred_decade: integer, e.g. 1990
  preferred_tags: list of strings
  vocals: "vocal" or "instrumental"
  target_duration: seconds
  preferred_occasions: list of strings
  language: string
  allow_explicit: false only if the user wants no explicit songs, otherwise null
  target_valence: number 0 (sad) to 1 (happy)
  target_tempo: BPM
  target_danceability: number 0-1
  target_instrumentalness: number 0-1

Reference:
{reference}

User said: {user_text}"""

        reply = self.ask(prompt, as_json=True, temperature=0)
        return parse_preferences_json(reply, vocab)


if __name__ == "__main__":
    client = GeminiClient()
    print(client.ask("Say hi in 5 words"))


# Thank you, https://github.com/codepath/ai110-module4tinker-docubot-solution/blob/main/llm_client.py
