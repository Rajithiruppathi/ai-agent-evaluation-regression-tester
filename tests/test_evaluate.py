import json
import os

import pytest

import evaluate
from evaluator import JudgeResult


def _fake_generate_response(question: str) -> str:
    return f"(mock response to) {question}"


def _make_fake_evaluate_response(verdicts: dict, reasons: dict | None = None):
    """verdicts maps question -> passed bool; reasons optionally maps question -> judge reason."""

    def _fake_evaluate_response(client, question, expected_answer, actual_response):
        passed = verdicts[question]
        reason = (reasons or {}).get(question, "mocked judge result")
        return JudgeResult(passed=passed, reason=reason)

    return _fake_evaluate_response


def _poison_generate_response(question: str) -> str:
    raise AssertionError("generate_response must not be called when validation fails")


def _poison_evaluate_response(client, question, expected_answer, actual_response):
    raise AssertionError("evaluate_response must not be called when validation fails")


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, verdicts, reasons=None, argv=None):
    _write_json(tmp_path / "test_cases.json", test_cases)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(evaluate, "generate_response", _fake_generate_response)
    monkeypatch.setattr(evaluate, "evaluate_response", _make_fake_evaluate_response(verdicts, reasons))
    return evaluate.main(argv=argv if argv is not None else [])


# --- default dataset behaviour (no --tests flag) ---

def test_default_dataset_is_used_when_no_tests_flag(tmp_path, monkeypatch, capsys):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True, "Q2?": True})

    results_path = tmp_path / "results" / "run_1.json"
    assert results_path.exists()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert all(r["passed"] for r in saved)

    output = capsys.readouterr().out
    assert "Total tests: 2" in output
    assert "Passed: 2" in output
    assert "Failed: 0" in output
    assert "Pass rate: 100.0%" in output
    assert "run_1.json" in output


def test_mixed_results_reports_correct_counts(tmp_path, monkeypatch, capsys):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True, "Q2?": False})

    output = capsys.readouterr().out
    assert "Total tests: 2" in output
    assert "Passed: 1" in output
    assert "Failed: 1" in output
    assert "Pass rate: 50.0%" in output


def test_run_evaluation_reuses_existing_result_schema(tmp_path, monkeypatch):
    test_cases = [{"id": "TC-001", "question": "Q1?", "expected_answer": "A1"}]
    _write_json(tmp_path / "test_cases.json", test_cases)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    monkeypatch.setattr(evaluate, "generate_response", _fake_generate_response)
    monkeypatch.setattr(evaluate, "evaluate_response", _make_fake_evaluate_response({"Q1?": True}))

    run_results, results_path = evaluate.run_evaluation()

    assert run_results == [
        {
            "test_id": "TC-001",
            "question": "Q1?",
            "expected_answer": "A1",
            "actual_response": "(mock response to) Q1?",
            "passed": True,
            "reason": "mocked judge result",
        }
    ]
    assert results_path == os.path.join("results", "run_1.json")


# --- report output: failures and reasons ---

def test_report_says_no_failures_when_all_pass(tmp_path, monkeypatch, capsys):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True, "Q2?": True})

    output = capsys.readouterr().out
    assert "No failures detected." in output
    assert "Failures:" not in output
    assert "Pass rate: 100.0%" in output


def test_report_lists_a_single_failure_with_reason(tmp_path, monkeypatch, capsys):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    _run_cli_with_verdicts(
        tmp_path,
        monkeypatch,
        test_cases,
        {"Q1?": True, "Q2?": False},
        {"Q2?": "Refund window is wrong."},
    )

    output = capsys.readouterr().out
    assert "Failures:" in output
    assert "TC-002 — FAIL" in output
    assert "Reason: Refund window is wrong." in output
    assert "TC-001 — FAIL" not in output
    assert "No failures detected." not in output


def test_report_lists_every_failed_test_with_its_reason(tmp_path, monkeypatch, capsys):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
        {"id": "TC-003", "question": "Q3?", "expected_answer": "A3"},
    ]
    _run_cli_with_verdicts(
        tmp_path,
        monkeypatch,
        test_cases,
        {"Q1?": False, "Q2?": True, "Q3?": False},
        {"Q1?": "Reason one.", "Q3?": "Reason three."},
    )

    output = capsys.readouterr().out
    assert "TC-001 — FAIL" in output
    assert "Reason: Reason one." in output
    assert "TC-003 — FAIL" in output
    assert "Reason: Reason three." in output
    assert "TC-002 — FAIL" not in output
    assert "Passed: 1" in output
    assert "Failed: 2" in output


def test_report_does_not_print_full_responses(tmp_path, monkeypatch, capsys):
    test_cases = [{"id": "TC-001", "question": "Q1?", "expected_answer": "A1"}]
    _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": False}, {"Q1?": "Wrong."})

    output = capsys.readouterr().out
    assert "(mock response to)" not in output
    assert "Result file:" in output


def test_report_header_and_result_file_line_are_present(tmp_path, monkeypatch, capsys):
    test_cases = [{"id": "TC-001", "question": "Q1?", "expected_answer": "A1"}]
    _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True})

    output = capsys.readouterr().out
    assert "Evaluation Report" in output
    assert "=================" in output
    assert os.path.join("results", "run_1.json") in output


# --- exit codes for CI ---

def test_all_pass_returns_exit_code_zero(tmp_path, monkeypatch):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    exit_code = _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True, "Q2?": True})

    assert exit_code == 0


def test_any_failure_returns_exit_code_one(tmp_path, monkeypatch):
    test_cases = [
        {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
        {"id": "TC-002", "question": "Q2?", "expected_answer": "A2"},
    ]
    exit_code = _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": True, "Q2?": False})

    assert exit_code == 1


def test_all_failures_return_exit_code_one(tmp_path, monkeypatch):
    test_cases = [{"id": "TC-001", "question": "Q1?", "expected_answer": "A1"}]
    exit_code = _run_cli_with_verdicts(tmp_path, monkeypatch, test_cases, {"Q1?": False})

    assert exit_code == 1


def test_unexpected_api_error_is_not_swallowed(tmp_path, monkeypatch):
    test_cases = [{"id": "TC-001", "question": "Q1?", "expected_answer": "A1"}]
    _write_json(tmp_path / "test_cases.json", test_cases)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(evaluate, "generate_response", _fake_generate_response)

    def _exploding_evaluate_response(client, question, expected_answer, actual_response):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr(evaluate, "evaluate_response", _exploding_evaluate_response)

    with pytest.raises(RuntimeError, match="simulated API outage"):
        evaluate.main(argv=[])


# --- custom dataset via --tests ---

def test_custom_test_file_is_actually_used(tmp_path, monkeypatch, capsys):
    # Deliberately do NOT write a test_cases.json here - if the code fell back to
    # the default path instead of the custom one, this test would fail loudly.
    custom_cases = [
        {"id": "CUSTOM-1", "question": "Custom question?", "expected_answer": "Custom answer"},
    ]
    custom_path = tmp_path / "my_tests.json"
    _write_json(custom_path, custom_cases)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    monkeypatch.setattr(evaluate, "generate_response", _fake_generate_response)
    monkeypatch.setattr(
        evaluate, "evaluate_response", _make_fake_evaluate_response({"Custom question?": True})
    )

    exit_code = evaluate.main(argv=["--tests", "my_tests.json"])

    assert exit_code == 0
    results_path = tmp_path / "results" / "run_1.json"
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved == [
        {
            "test_id": "CUSTOM-1",
            "question": "Custom question?",
            "expected_answer": "Custom answer",
            "actual_response": "(mock response to) Custom question?",
            "passed": True,
            "reason": "mocked judge result",
        }
    ]

    output = capsys.readouterr().out
    assert "Total tests: 1" in output


# --- validation failures: clear error, non-zero exit, no API call ---

def _poison_api_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(evaluate, "generate_response", _poison_generate_response)
    monkeypatch.setattr(evaluate, "evaluate_response", _poison_evaluate_response)


def test_missing_file_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "does_not_exist.json"])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_invalid_json_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "broken.json"])

    assert exit_code == 1
    assert "not valid json" in capsys.readouterr().err.lower()


def test_non_list_top_level_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    _write_json(tmp_path / "not_a_list.json", {"id": "TC-001"})
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "not_a_list.json"])

    assert exit_code == 1
    assert "json list" in capsys.readouterr().err.lower()


def test_missing_required_field_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    _write_json(tmp_path / "missing_field.json", [{"id": "TC-001", "question": "Q1?"}])
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "missing_field.json"])

    assert exit_code == 1
    assert "expected_answer" in capsys.readouterr().err


def test_non_string_field_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    _write_json(
        tmp_path / "bad_type.json",
        [{"id": "TC-001", "question": "Q1?", "expected_answer": 12345}],
    )
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "bad_type.json"])

    assert exit_code == 1
    assert "must be a string" in capsys.readouterr().err.lower()


def test_empty_dataset_returns_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    _write_json(tmp_path / "empty.json", [])
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "empty.json"])

    assert exit_code == 1
    assert "no test cases" in capsys.readouterr().err.lower()


def test_duplicate_test_ids_return_nonzero_without_api_call(tmp_path, monkeypatch, capsys):
    _write_json(
        tmp_path / "duplicates.json",
        [
            {"id": "TC-001", "question": "Q1?", "expected_answer": "A1"},
            {"id": "TC-001", "question": "Q2?", "expected_answer": "A2"},
        ],
    )
    monkeypatch.chdir(tmp_path)
    _poison_api_calls(monkeypatch)

    exit_code = evaluate.main(argv=["--tests", "duplicates.json"])

    assert exit_code == 1
    assert "duplicate" in capsys.readouterr().err.lower()
