import json
import sys

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def load_results(path: str) -> dict:
    """Load a results JSON file and index it by test_id."""
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r["test_id"]: r for r in results}


def status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def is_regression(previous_passed: bool, current_passed: bool) -> bool:
    """A regression is specifically: PASS in the previous run -> FAIL in the new run."""
    return bool(previous_passed) and not current_passed


def compare_runs(previous_path: str, current_path: str) -> None:
    previous = load_results(previous_path)
    current = load_results(current_path)

    all_test_ids = sorted(set(previous) | set(current))
    regressions = 0

    for test_id in all_test_ids:
        prev = previous.get(test_id)
        curr = current.get(test_id)

        print("-" * 60)
        print(f"Test ID: {test_id}")

        if prev is None:
            assert curr is not None  # test_id came from `current`, since it's not in `previous`
            print(f"Question: {curr['question']}")
            print("Previous result: (no data - test not present in previous run)")
            print(f"Current result: {status(curr['passed'])}")
            print("Regression: N/A")
            print(f"Reason: {curr['reason']}")
            continue

        if curr is None:
            print(f"Question: {prev['question']}")
            print(f"Previous result: {status(prev['passed'])}")
            print("Current result: (no data - test not present in current run)")
            print("Regression: N/A")
            continue

        regression = is_regression(prev["passed"], curr["passed"])
        if regression:
            regressions += 1

        print(f"Question: {curr['question']}")
        print(f"Previous result: {status(prev['passed'])}")
        print(f"Current result: {status(curr['passed'])}")
        print(f"Regression: {'YES' if regression else 'no'}")
        print(f"Reason: {curr['reason']}")

    prev_total = len(previous)
    prev_passed = sum(1 for r in previous.values() if r["passed"])
    curr_total = len(current)
    curr_passed = sum(1 for r in current.values() if r["passed"])

    print("-" * 60)
    print("Summary")
    print(f"Previous run: {prev_passed}/{prev_total} passed")
    print(f"Current run: {curr_passed}/{curr_total} passed")
    print(f"Regressions detected: {regressions}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python regression.py <previous_run.json> <current_run.json>")
        sys.exit(1)

    compare_runs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
