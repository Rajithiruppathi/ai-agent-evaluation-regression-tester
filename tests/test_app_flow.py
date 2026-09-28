import json

import app
from evaluator import JudgeResult


def _fake_generate_response(question: str) -> str:
    return f"(mock response to) {question}"


def _fake_evaluate_response(client, question, expected_answer, actual_response):
    return JudgeResult(passed=True, reason="mocked judge result")


def test_main_runs_end_to_end_without_a_real_api_call(tmp_path, monkeypatch):
    # Isolated working directory with its own tiny test_cases.json
    test_cases = [
        {"id": "TC-001", "question": "What is your refund policy?", "expected_answer": "Full refund in 30 days."},
    ]
    (tmp_path / "test_cases.json").write_text(json.dumps(test_cases), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # A fake key is enough - the real client is never actually called, because
    # both integration points below (generate_response, evaluate_response) are mocked.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    monkeypatch.setattr(app, "generate_response", _fake_generate_response)
    monkeypatch.setattr(app, "evaluate_response", _fake_evaluate_response)

    app.main()

    results_path = tmp_path / "results" / "run_1.json"
    assert results_path.exists()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved == [
        {
            "test_id": "TC-001",
            "question": "What is your refund policy?",
            "expected_answer": "Full refund in 30 days.",
            "actual_response": "(mock response to) What is your refund policy?",
            "passed": True,
            "reason": "mocked judge result",
        }
    ]
