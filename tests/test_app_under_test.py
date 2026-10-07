import pytest

import app_under_test
import evaluator


class _FakeThinkingBlock:
    type = "thinking"
    thinking = "internal reasoning"


class _ThinkingOnlyMessages:
    def create(self, **kwargs):
        return type("Response", (), {"content": [_FakeThinkingBlock()]})()


class _ThinkingOnlyClient:
    messages = _ThinkingOnlyMessages()


def test_generate_response_raises_clear_error_when_only_thinking_block(monkeypatch):
    monkeypatch.setattr(app_under_test, "_get_client", lambda: _ThinkingOnlyClient())

    with pytest.raises(RuntimeError, match="contained no text block"):
        app_under_test.generate_response("Any question?")


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
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._raw_text)


class _FakeClient:
    def __init__(self, raw_text):
        self.messages = _FakeMessages(raw_text)


def test_generate_response_returns_text_from_the_client(monkeypatch):
    fake_client = _FakeClient("This is a mocked answer.")
    monkeypatch.setattr(app_under_test, "_get_client", lambda: fake_client)

    result = app_under_test.generate_response("Any question?")

    assert result == "This is a mocked answer."


def test_generate_response_includes_knowledge_base_in_the_prompt(monkeypatch):
    fake_client = _FakeClient("This is a mocked answer.")
    monkeypatch.setattr(app_under_test, "_get_client", lambda: fake_client)

    app_under_test.generate_response("What is your refund policy?")

    system_prompt = fake_client.messages.last_kwargs["system"]
    assert "Refunds" in system_prompt
    assert "30 days" in system_prompt


def test_knowledge_base_loads_expected_topics():
    knowledge = app_under_test._load_knowledge_base()

    assert "refund_policy" in knowledge
    assert "account_cancellation" in knowledge
    assert "shipping_times" in knowledge
    assert "30 days" in knowledge["refund_policy"]["policy"]
    assert "Cancel Account" in knowledge["account_cancellation"]["policy"]
    assert "3 to 5 business days" in knowledge["shipping_times"]["policy"]


def test_evaluator_does_not_depend_on_app_under_test():
    """evaluator.py must only know about (question, expected_answer, actual_response) -
    it should have no import of, or attribute from, app_under_test."""
    assert not hasattr(evaluator, "app_under_test")
    assert not hasattr(evaluator, "generate_response")

    with open("evaluator.py", "r", encoding="utf-8") as f:
        source = f.read()
    assert "app_under_test" not in source
