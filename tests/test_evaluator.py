import json

import pytest
from pydantic import ValidationError

from evaluator import JudgeResult, evaluate_response


# --- JudgeResult structure validation (no API calls involved) ---

def test_judge_result_accepts_valid_data():
    result = JudgeResult(passed=True, reason="Matches the expected answer.")
    assert result.passed is True
    assert isinstance(result.passed, bool)
    assert isinstance(result.reason, str)


def test_judge_result_rejects_non_boolean_passed():
    with pytest.raises(ValidationError):
        JudgeResult(passed="banana", reason="not a real boolean value")


def test_judge_result_rejects_non_string_reason():
    with pytest.raises(ValidationError):
        JudgeResult(passed=True, reason=12345)


def test_judge_result_rejects_missing_fields():
    with pytest.raises(ValidationError):
        JudgeResult(passed=True)  # missing "reason"


# --- evaluate_response: uses a fake Anthropic client, never a real API call ---

class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, raw_text):
        self._raw_text = raw_text

    def create(self, **kwargs):
        return _FakeResponse(self._raw_text)


class _FakeClient:
    def __init__(self, raw_text):
        self.messages = _FakeMessages(raw_text)


def test_evaluate_response_parses_valid_judge_output():
    fake_client = _FakeClient('{"passed": true, "reason": "Meaning matches."}')

    result = evaluate_response(
        fake_client,
        question="What is your refund policy?",
        expected_answer="Full refund within 30 days.",
        actual_response="You can get a full refund if you ask within 30 days.",
    )

    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert result.reason == "Meaning matches."


def _judge(raw_text):
    return evaluate_response(
        _FakeClient(raw_text),
        question="Irrelevant",
        expected_answer="Irrelevant",
        actual_response="Irrelevant",
    )


def test_evaluate_response_rejects_invalid_structured_output():
    # "passed" is not a valid boolean value -> pydantic should reject it
    with pytest.raises(ValidationError):
        _judge('{"passed": "banana", "reason": "bad output"}')


def test_evaluate_response_accepts_json_fenced_output():
    result = _judge('```json\n{"passed": true, "reason": "Fenced JSON."}\n```')

    assert result.passed is True
    assert result.reason == "Fenced JSON."


def test_evaluate_response_accepts_plain_fenced_output():
    result = _judge('```\n{"passed": false, "reason": "Plain fence."}\n```')

    assert result.passed is False
    assert result.reason == "Plain fence."


def test_evaluate_response_rejects_non_json_text():
    with pytest.raises(json.JSONDecodeError):
        _judge("This is not JSON at all.")


def test_evaluate_response_does_not_extract_json_from_surrounding_prose():
    with pytest.raises(json.JSONDecodeError):
        _judge('Here is my verdict:\n```json\n{"passed": true, "reason": "x"}\n```\nThanks!')
