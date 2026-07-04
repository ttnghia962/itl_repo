from app.config import _extract_groq_api_keys, Settings


def test_settings_default_chunk_sizes():
    settings = Settings()
    assert settings.chunk_max_tokens == 200
    assert settings.chunk_overlap_tokens == 20


def test_extract_groq_api_keys_sorts_by_numeric_suffix():
    env = {
        "GROQ_API_KEY2": "second",
        "GROQ_API_KEY10": "tenth",
        "GROQ_API_KEY1": "first",
    }
    assert _extract_groq_api_keys(env) == ["first", "second", "tenth"]


def test_extract_groq_api_keys_ignores_non_matching_and_empty_values():
    env = {
        "GROQ_API_KEY1": "first",
        "GROQ_API_KEY_EXTRA": "should-not-match",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "GROQ_API_KEY2": "",
        "GROQ_API_KEY3": None,
    }
    assert _extract_groq_api_keys(env) == ["first"]


def test_extract_groq_api_keys_returns_empty_list_when_none_present():
    assert _extract_groq_api_keys({"UNRELATED": "value"}) == []
