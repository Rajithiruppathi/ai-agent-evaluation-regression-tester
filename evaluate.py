"""Command-line entry point: run the evaluation suite and print a report.

    python evaluate.py                       # uses test_cases.json
    python evaluate.py --tests my_tests.json

Exit code is 0 when every test passes, 1 when any test fails or the dataset is invalid.
"""

import argparse
import glob
import json
import os
import re
import sys

import anthropic
from dotenv import load_dotenv

from app_under_test import generate_response
from evaluator import evaluate_response

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

REQUIRED_FIELDS = ("id", "question", "expected_answer")


class DatasetError(ValueError):
    """The test-case dataset is missing or malformed."""


def validate_test_cases(data) -> None:
    """Raise DatasetError with a clear message if `data` doesn't match the expected
    test-case schema: a non-empty JSON list of {id, question, expected_answer} objects
    with unique string ids."""
    if not isinstance(data, list):
        raise DatasetError("Test file must contain a JSON list of test cases.")

    if not data:
        raise DatasetError("Test file contains no test cases.")

    seen_ids = set()

    for index, case in enumerate(data):
        if not isinstance(case, dict):
            raise DatasetError(f"Test case at index {index} must be a JSON object.")

        for field in REQUIRED_FIELDS:
            if field not in case:
                raise DatasetError(f"Test case at index {index} is missing required field '{field}'.")
            if not isinstance(case[field], str):
                raise DatasetError(f"Test case at index {index} field '{field}' must be a string.")

        if case["id"] in seen_ids:
            raise DatasetError(f"Duplicate test ID found: '{case['id']}'.")
        seen_ids.add(case["id"])


def load_and_validate_test_cases(path: str) -> list[dict]:
    """Load a test-case JSON file and validate it against the expected schema.
    Raises DatasetError with a human-readable message on any problem."""
    if not os.path.isfile(path):
        raise DatasetError(f"Test file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetError(f"Test file is not valid JSON: {path} ({e})") from e

    validate_test_cases(data)
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Agent Evaluation & Regression Tester.")
    parser.add_argument(
        "--tests",
        dest="tests_path",
        default="test_cases.json",
        help="Path to a JSON test-case dataset (default: test_cases.json)",
    )
    return parser.parse_args(argv)


def get_next_run_path(results_dir: str = "results") -> str:
    """Return a path like results/run_1.json, results/run_2.json, ... picking the next free number."""
    os.makedirs(results_dir, exist_ok=True)

    existing_numbers = []
    for path in glob.glob(os.path.join(results_dir, "run_*.json")):
        match = re.search(r"run_(\d+)\.json$", path)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(results_dir, f"run_{next_number}.json")


def save_results(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def get_judge_client() -> anthropic.Anthropic:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def run_test_case(judge_client, case: dict) -> dict:
    """Generate a response for one test case and evaluate it. Returns a result dict
    in the standard results/run_N.json shape."""
    actual_response = generate_response(case["question"])
    result = evaluate_response(
        judge_client,
        question=case["question"],
        expected_answer=case["expected_answer"],
        actual_response=actual_response,
    )
    return {
        "test_id": case["id"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "actual_response": actual_response,
        "passed": result.passed,
        "reason": result.reason,
    }


def run_evaluation(tests_path: str = "test_cases.json") -> tuple[list[dict], str]:
    test_cases = load_and_validate_test_cases(tests_path)

    judge_client = get_judge_client()
    run_results = [run_test_case(judge_client, case) for case in test_cases]

    results_path = get_next_run_path()
    save_results(run_results, results_path)

    return run_results, results_path


def print_summary(run_results: list[dict], results_path: str) -> None:
    total = len(run_results)
    passed = sum(1 for r in run_results if r["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0

    print("Evaluation Report")
    print("=================")
    print()
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print()

    if failed:
        print("Failures:")
        for r in run_results:
            if not r["passed"]:
                print(f"  {r['test_id']} — FAIL")
                print(f"  Reason: {r['reason']}")
    else:
        print("No failures detected.")

    print()
    print(f"Result file: {results_path}")


def main(argv: list[str] | None = None) -> int:
    """Return 0 if every test passes, 1 if any test fails or the dataset is invalid.
    Unexpected errors (e.g. API failures) propagate, so Python exits non-zero."""
    args = parse_args(argv)

    try:
        run_results, results_path = run_evaluation(args.tests_path)
    except DatasetError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_summary(run_results, results_path)
    return 0 if all(r["passed"] for r in run_results) else 1


if __name__ == "__main__":
    sys.exit(main())
