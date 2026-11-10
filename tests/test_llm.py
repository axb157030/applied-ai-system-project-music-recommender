import pytest
from src.llm import GeminiClient, LLMError

def test_gemini_client_ask():
    # This test assumes that the GEMINI_API_KEY is set in the environment.
    # It will fail if the key is missing or invalid.
    client = GeminiClient()
    response = client.ask("What is 2 + 2?")
    assert "4" in response

def test_gemini_client_ask_as_json():
    client = GeminiClient()
    response = client.ask('{"question": "What is the capital of France?"}', as_json=True)
    assert isinstance(response, str)  # The response should be a JSON string
    assert "Paris" in response  # The answer should mention Paris

def test_gemini_client_ask_invalid_key(monkeypatch):
    # Temporarily unset the GEMINI_API_KEY to simulate a missing key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiClient()

def test_gemini_client_ask_and_llm_error_exception_caught_when_client_raises_exception(monkeypatch):
    # Simulate a network error by monkeypatching the ask method to raise an exception
    client = GeminiClient()

    def mock_ask(*args, **kwargs):
        raise Exception("Simulated network error")

    client.client.models.generate_content = mock_ask
    with pytest.raises(LLMError):
        client.ask("What is 2 + 2?")
    

# ---------- RAG and extraction (fake ask, so no network) ----------

def _client_with_fake_ask(monkeypatch, reply):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    client = GeminiClient()
    calls = []

    def fake_ask(prompt, as_json=False, temperature=None):
        calls.append(prompt)
        return reply

    client.ask = fake_ask
    return client, calls


def test_answer_from_snippets_refuses_without_calling_gemini(monkeypatch):
    client, calls = _client_with_fake_ask(monkeypatch, "should not be used")
    assert client.answer_from_snippets("what is energy?", []) == "I do not know based on the docs I have."
    assert calls == []


def test_answer_from_snippets_puts_snippets_in_the_prompt(monkeypatch):
    client, calls = _client_with_fake_ask(monkeypatch, "ok")
    client.answer_from_snippets("what is energy?", [("attributes.md", "Energy is intensity.")])
    assert "attributes.md" in calls[0] and "Energy is intensity." in calls[0]


def test_extract_preferences_validates_model_output(monkeypatch):
    # The model "returns" a good value, a wild value and an unknown key.
    client, _ = _client_with_fake_ask(monkeypatch, '{"target_energy": 0.2, "target_valence": 9, "evil": 1}')
    assert client.extract_preferences("calm", [("attributes.md", "x")]) == {"target_energy": 0.2}


def test_extract_preferences_bad_json_gives_empty(monkeypatch):
    client, _ = _client_with_fake_ask(monkeypatch, "not json")
    assert client.extract_preferences("calm", [("attributes.md", "x")]) == {}
