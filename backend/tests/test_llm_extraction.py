import json
from types import SimpleNamespace

import pytest

from app.services import llm_extraction
from app.services.llm_extraction import extract_cv_fields, get_groq_client


class FakeClient:
    def __init__(self, response_json=None, raw_content=None, raise_error=None):
        self._response_json = response_json
        self._raw_content = raw_content
        self._raise_error = raise_error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self._raise_error:
            raise self._raise_error

        if self._raw_content is not None:
            content = self._raw_content
        else:
            content = json.dumps(self._response_json)

        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_cv_fields_parses_required_fields():
    fake = FakeClient(
        {
            "name": "John Smith",
            "email": "john@example.com",
            "phone": "123-456",
            "skills": ["Python", "Machine Learning"],
            "education": [],
            "experience": [],
            "current_role": "Data Scientist",
            "domain": "Information Technology",
            "extra_fields": {"certifications": "AWS Certified Solutions Architect"},
        }
    )
    result = extract_cv_fields("dummy resume text", client=fake)
    assert result.name == "John Smith"
    assert result.skills == ["Python", "Machine Learning"]
    assert result.extra_fields["certifications"] == "AWS Certified Solutions Architect"


def test_extract_cv_fields_tolerates_missing_optional_keys():
    fake = FakeClient({"name": "Jane"})
    result = extract_cv_fields("dummy", client=fake)
    assert result.name == "Jane"
    assert result.skills == []
    assert result.extra_fields == {}


def test_extract_cv_fields_wraps_groq_api_error():
    fake = FakeClient(raise_error=RuntimeError("Network error"))
    with pytest.raises(RuntimeError) as exc_info:
        extract_cv_fields("dummy", client=fake)
    assert "Groq CV extraction API call failed" in str(exc_info.value)
    assert "Network error" in str(exc_info.value.__cause__)


def test_extract_cv_fields_wraps_json_parsing_error():
    fake = FakeClient(raw_content="not json")
    with pytest.raises(ValueError) as exc_info:
        extract_cv_fields("dummy", client=fake)
    assert "Groq returned invalid JSON for CV extraction" in str(exc_info.value)


def test_extract_cv_fields_drops_non_string_extra_field_values():
    fake = FakeClient(
        {
            "name": "Jane",
            "extra_fields": {"certifications": "AWS Certified", "languages": None, "projects": None},
        }
    )
    result = extract_cv_fields("dummy", client=fake)
    assert result.extra_fields == {"certifications": "AWS Certified"}


def test_extract_cv_fields_wraps_schema_validation_error():
    # skills expects List[str], but we provide a string
    fake = FakeClient({"name": "John", "skills": "not-a-list-but-a-string"})
    with pytest.raises(ValueError) as exc_info:
        extract_cv_fields("dummy", client=fake)
    assert "Groq CV extraction response did not match expected schema" in str(exc_info.value)


class FakeGroq:
    def __init__(self, api_key):
        self.api_key = api_key


def test_get_groq_client_round_robins_across_configured_keys(monkeypatch):
    monkeypatch.setattr(llm_extraction, "_client_cycle", None)
    monkeypatch.setattr(llm_extraction.settings, "groq_api_keys", ["key-a", "key-b"])
    monkeypatch.setattr(llm_extraction, "Groq", FakeGroq)

    first = get_groq_client()
    second = get_groq_client()
    third = get_groq_client()

    assert [first.api_key, second.api_key, third.api_key] == ["key-a", "key-b", "key-a"]


def test_get_groq_client_raises_clear_error_when_no_keys_configured(monkeypatch):
    monkeypatch.setattr(llm_extraction, "_client_cycle", None)
    monkeypatch.setattr(llm_extraction.settings, "groq_api_keys", [])
    monkeypatch.setattr(llm_extraction, "Groq", FakeGroq)

    with pytest.raises(RuntimeError) as exc_info:
        get_groq_client()
    assert "No Groq API keys configured" in str(exc_info.value)
