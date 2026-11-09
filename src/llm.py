"""
Gemini client wrapper.

Reads the API key from the GEMINI_API_KEY environment variable and offers two ways
to ask a question:
- naive_answer_over_full_docs: no retrieval, the model answers from general knowledge
  (baseline for comparison)
- answer_from_snippets: RAG style, the model may only use the snippets it is given

Tunable: prompt wording, the refusal rule, and how strictly the model sticks to the snippets.
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()   # reads GEMINI_API_KEY from your .env file


# Model used for every request. Change it here to switch models.
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"


 
class LLMError(Exception):
    """Raised when a Gemini request fails (network error, quota, invalid key, ...)."""


class GeminiClient:
    """
    Thin wrapper around the Gemini model.

    Usage:
        client = GeminiClient()
        answer = client.naive_answer_over_full_docs(query, all_text)
        # or
        answer = client.answer_from_snippets(query, snippets)
    """


    def __init__(self):
        # Fail early with a clear message if the key is missing.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )

        self.client = genai.Client(api_key=api_key)

    def ask(self, prompt, as_json=False):
        """Send one prompt to Gemini and return the reply text. Retries once."""
        config = None
        if as_json:
            config = types.GenerateContentConfig(response_mime_type="application/json")

        last_error = None
        for attempt in range(2):
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

    def naive_answer_over_full_docs(self, query, all_text):
        """Answer from the model's general knowledge. Only the question is sent."""
        # all_text is deliberately ignored: this is the no-retrieval baseline.
        prompt = f"""
    You are a documentation assistant. 
    Answer this developer question: {query}
    """
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt
            )
            return (response.text or "").strip()
        except Exception as e:
            # Return the error as text so a failed API call does not crash the caller.
            return f"Unable to generate an answer. ({type(e).__name__}: {e})"

    # -----------------------------------------------------------
    # RAG: generation grounded in retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples chosen by the retriever.

        The prompt shows each snippet with its filename, tells the model to rely only
        on them, and requires a fixed "I do not know" reply when they are not enough.
        """

        # Nothing was retrieved: refuse without spending an API call.
        if not snippets:
            return "I do not know based on the docs I have."

        # Label each snippet with its filename so the model can cite its sources.
        context_blocks = []
        for filename, text in snippets:
            block = f"File: {filename}\n{text}\n"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        # The rules at the end keep the answer grounded: use only the snippets, refuse
        # with one exact sentence, and name the files that were used.
        prompt = f"""
You are a cautious documentation assistant helping developers understand a codebase.

You will receive:
- A developer question
- A small set of snippets from project files

Your job:
- Answer the question using only the information in the snippets.
- If the snippets do not provide enough evidence, refuse to guess.

Snippets:
{context}

Developer question:
{query}

Rules:
- Use only the information in the snippets. Do not invent new functions,
  endpoints, or configuration values.
- If the snippets are not enough to answer confidently, reply exactly:
  "I do not know based on the docs I have."
- When you do answer, briefly mention which files you relied on.
"""
if __name__ == "__main__":
    client = GeminiClient()
    print(client.ask("Say hi in 5 words"))
