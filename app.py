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


def load_test_cases(path: str = "test_cases.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    print("AI Agent Evaluation & Regression Tester")

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    judge_client = anthropic.Anthropic(api_key=api_key)

    test_cases = load_test_cases()
    run_results = []

    for case in test_cases:
        actual_response = generate_response(case["question"])
        result = evaluate_response(
            judge_client,
            question=case["question"],
            expected_answer=case["expected_answer"],
            actual_response=actual_response,
        )

        print("-" * 60)
        print(f"Test ID: {case['id']}")
        print(f"Question: {case['question']}")
        print(f"Expected answer: {case['expected_answer']}")
        print(f"Actual response: {actual_response}")
        print("PASS" if result.passed else "FAIL")
        print(f"Reason: {result.reason}")

        run_results.append(
            {
                "test_id": case["id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "actual_response": actual_response,
                "passed": result.passed,
                "reason": result.reason,
            }
        )

    results_path = get_next_run_path()
    save_results(run_results, results_path)
    print("-" * 60)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
