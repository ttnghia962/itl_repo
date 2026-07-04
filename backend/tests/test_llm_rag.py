import json
from types import SimpleNamespace

import pytest

from app.services.llm_rag import generate_answer


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


def test_generate_answer_names_best_candidate():
    fake = FakeClient(
        {
            "answer": "Alice is the best fit because she has 6 years of Python and ML experience.",
            "best_candidate_id": "cv1",
            "best_candidate_reason": "Alice has 6 years of hands-on Python and Machine Learning experience.",
        }
    )
    contexts = [
        {"candidate_id": "cv1", "name": "Alice", "skills": ["Python", "Machine Learning"]},
        {"candidate_id": "cv2", "name": "Bob", "skills": ["Accounting"]},
    ]
    result = generate_answer("Who has Python and Machine Learning experience?", contexts, client=fake)
    assert result["best_candidate_id"] == "cv1"
    assert "Alice" in result["answer"]
    assert result["best_candidate_reason"] == "Alice has 6 years of hands-on Python and Machine Learning experience."


def test_generate_answer_defaults_reason_to_none_when_missing():
    fake = FakeClient({"answer": "Alice is the best fit.", "best_candidate_id": "cv1"})
    result = generate_answer("Who has Python?", [{"candidate_id": "cv1", "name": "Alice"}], client=fake)
    assert result["best_candidate_reason"] is None


def test_generate_answer_handles_no_match():
    fake = FakeClient({"answer": "No candidate matches.", "best_candidate_id": None})
    result = generate_answer("Who knows quantum computing?", [], client=fake)
    assert result["best_candidate_id"] is None


def test_generate_answer_wraps_groq_api_error():
    fake = FakeClient(raise_error=RuntimeError("Network error"))
    with pytest.raises(RuntimeError) as exc_info:
        generate_answer("Who has Python?", [], client=fake)
    assert "Groq RAG answer generation API call failed" in str(exc_info.value)
    assert "Network error" in str(exc_info.value.__cause__)


def test_generate_answer_wraps_json_parsing_error():
    fake = FakeClient(raw_content="not json")
    with pytest.raises(ValueError) as exc_info:
        generate_answer("Who has Python?", [], client=fake)
    assert "Groq returned invalid JSON for RAG answer generation" in str(exc_info.value)
