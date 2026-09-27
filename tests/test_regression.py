import json

from regression import compare_runs, is_regression


# --- is_regression: the four required transitions ---

def test_pass_to_pass_is_not_a_regression():
    assert is_regression(previous_passed=True, current_passed=True) is False


def test_pass_to_fail_is_a_regression():
    assert is_regression(previous_passed=True, current_passed=False) is True


def test_fail_to_pass_is_not_a_regression():
    assert is_regression(previous_passed=False, current_passed=True) is False


def test_fail_to_fail_is_not_a_regression():
    assert is_regression(previous_passed=False, current_passed=False) is False


# --- compare_runs: end-to-end over fixture files, no API calls involved ---

def _make_result(test_id, passed, reason="because"):
    return {
        "test_id": test_id,
        "question": f"Question for {test_id}",
        "expected_answer": "expected",
        "actual_response": "actual",
        "passed": passed,
        "reason": reason,
    }


def test_compare_runs_reports_one_regression(tmp_path, capsys):
    previous = [
        _make_result("TC-001", passed=True),
        _make_result("TC-002", passed=False),
        _make_result("TC-003", passed=True),
    ]
    current = [
        _make_result("TC-001", passed=False, reason="now contradicts expected answer"),
        _make_result("TC-002", passed=True),
        _make_result("TC-003", passed=True),
    ]

    previous_path = tmp_path / "previous.json"
    current_path = tmp_path / "current.json"
    previous_path.write_text(json.dumps(previous), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    compare_runs(str(previous_path), str(current_path))

    output = capsys.readouterr().out
    assert "Regressions detected: 1" in output
    assert "Previous run: 2/3 passed" in output
    assert "Current run: 2/3 passed" in output
