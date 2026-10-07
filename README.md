# AI Agent Evaluation & Regression Tester

A small Python tool that runs test questions against an LLM application, uses an LLM judge to check each answer against an expected answer, saves the results as JSON, and compares two saved runs to detect regressions.

## The problem

LLM applications can regress when prompts, models, instructions, knowledge sources, or application code change. A change that improves one answer can quietly break another. Traditional tests based on exact string matching are often not enough: an answer can be semantically correct while using different wording, and a string test will reject it. Evaluation needs to judge meaning, and it needs to compare results over time so that a previously passing behaviour that now fails is visible.

## What this project does

1. It runs every test question through the application under test.
2. It asks an LLM judge whether each answer matches the expected answer in meaning.
3. It saves the results to `results/run_N.json`.
4. It compares two saved runs and reports any test that went from passing to failing.

For example:

```
Previous run:
  TC-001 → PASS

New run:
  TC-001 → FAIL

Regression detected.
```

## Architecture

```
evaluate.py  (CLI)
    │
    ├── loads and validates the test dataset (test_cases.json or --tests file)
    │
    ├── for each test case:
    │     ├── app_under_test.generate_response(question)
    │     │       └── Anthropic API, with knowledge_base.json in the prompt
    │     │
    │     └── evaluator.evaluate_response(...)
    │             └── Anthropic API (LLM judge) → JudgeResult(passed, reason)
    │
    ├── prints the evaluation report
    └── saves results/run_N.json   (exit code 0 if all passed, otherwise 1)

regression.py  (separate CLI)
    └── compares two saved run files → PASS → FAIL regressions
```

## Key features

- **Configurable test datasets**: use the default `test_cases.json` or pass your own file with `--tests`.
- **Semantic LLM-as-judge evaluation**: the judge compares meaning, not exact wording.
- **Structured judge result**: the judge's answer is validated with a Pydantic model (`passed: bool`, `reason: str`).
- **Clear PASS / FAIL output**: each failure is listed with the judge's reason.
- **Saved JSON results**: every run is written to `results/run_N.json`.
- **Regression comparison**: `regression.py` reports tests that went from PASS to FAIL between two runs.
- **Validation before API calls**: malformed or empty datasets are rejected before any request is sent.
- **CI-friendly exit codes**: the CLI returns a meaningful status code.
- **Fenced JSON handling**: judge responses wrapped in a Markdown code fence are parsed correctly.
- **Automated tests**: 39 pytest tests, none of which call the real API.

## Project structure

| Path | Purpose |
|---|---|
| `evaluate.py` | The CLI entry point. Loads and validates the dataset, runs the evaluation, prints the report, saves the results, and sets the exit code. |
| `app_under_test.py` | The AI application being evaluated. Exposes `generate_response(question) -> str`, which answers using `knowledge_base.json` and Claude. Replace this module to evaluate a different application. |
| `evaluator.py` | The LLM judge. Sends the question, expected answer, and actual response to Claude and returns a validated `JudgeResult`. |
| `regression.py` | Compares two saved run files and reports PASS → FAIL regressions. It doesn't call any API. |
| `test_cases.json` | The default evaluation dataset. |
| `knowledge_base.json` | Fictional customer-support policies used by the sample application. |
| `examples/` | Saved evaluation results, including the regression demonstration. See [Demo data](#demo-data). |
| `tests/` | The pytest suite. |
| `results/` | Generated run files. Git-ignored. |

## How it works

1. `evaluate.py` reads the dataset (`test_cases.json` by default, or the file given with `--tests`) and validates it.
2. For each test case, `app_under_test.generate_response()` sends the question to Claude, along with the policies from `knowledge_base.json`, and returns the answer.
3. `evaluator.evaluate_response()` sends the question, the expected answer, and the actual answer to Claude as the judge. The judge replies with a PASS/FAIL verdict and a short reason, which is validated against the `JudgeResult` model.
4. Results are collected and saved to the next free `results/run_N.json` file.
5. The report shows the totals, the pass rate, the reasons for any failures, and the path of the saved file.
6. `regression.py`, run separately, compares two saved run files by `test_id` and reports tests that went from PASS to FAIL.

## Quick start

These commands are for Windows PowerShell or Command Prompt, from the project folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a file named `.env` in the project folder with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

Replace `your_key_here` with your own key. Never commit a real key or paste one into the README, code, or tests. `.env` is listed in `.gitignore`.

## Usage

Run the default dataset:

```powershell
python evaluate.py
```

Run your own dataset:

```powershell
python evaluate.py --tests my_tests.json
```

Compare two saved runs:

```powershell
python regression.py examples/regression_demo/previous_run_all_pass.json examples/regression_demo/current_run_refund_regression.json
```

### Custom dataset format

A dataset is a JSON list of test cases. Each case needs an `id`, a `question`, and an `expected_answer`, all as strings. IDs must be unique.

```json
[
  {
    "id": "TC-001",
    "question": "What is your refund policy?",
    "expected_answer": "Full refund within 30 days if the item is unused."
  }
]
```

The dataset is validated before any API call. The file must exist, contain valid JSON, be a non-empty list, have the required fields as strings, and have unique IDs. If validation fails, the CLI prints a message such as `Error: Duplicate test ID found: 'TC-001'.` and exits with code 1 without contacting the API.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Every test passed. |
| `1` | One or more tests failed, or the dataset failed validation. |
| `2` | Invalid command-line arguments (reported by the argument parser). |

These codes let a script or CI job treat the evaluation as a quality gate.

## Regression comparison

`regression.py` takes two saved run files and reports, for each test, the previous result, the current result, and whether the test regressed (PASS → FAIL). It finishes with a summary:

```
Summary
Previous run: 3/3 passed
Current run: 2/3 passed
Regressions detected: 1
```

`regression.py` prints this report and exits with code `0` even when it finds regressions, so it isn't yet a CI gate on its own. Use its output, or add a check to your own script, if you need to fail a build on a regression.

## Demo data

All data in this project is fictional.

- `knowledge_base.json` contains fictional customer-support policies. It doesn't describe any real company's policies.
- `examples/regression_demo/previous_run_all_pass.json` is a saved run in which all three tests passed, using the 30-day refund policy.
- `examples/regression_demo/current_run_refund_regression.json` is a saved run in which the refund test fails. To produce it, the refund policy in `knowledge_base.json` was deliberately changed from 30 days to 14 days. That incorrect policy was used only to generate this demonstration and was then reverted.
- `examples/baseline_no_knowledge_base.json` is a saved run from before the knowledge base was added. All three tests fail, because the application had no policy information to answer from. It shows why the application needs ground truth to answer the test questions.

Running the regression command above reproduces the intended result: `TC-001` regresses from PASS to FAIL, and the other two tests stay PASS.

## Testing

```powershell
pytest
```

The suite contains 39 tests. They don't call the Anthropic API. They use fake clients and mocks for the application and the judge, and they run the CLI against temporary datasets.

## CI / GitHub Actions

The repository includes a GitHub Actions workflow, `.github/workflows/claude-review.yml`. It runs on pull requests to `master`, sends the diff to Claude, and posts a code review as a PR comment. It doesn't run the test suite or the evaluation.

The CLI's exit codes make it suitable for a CI quality gate. A job that runs `python evaluate.py` would fail on any test failure, on invalid input, or on invalid arguments. Running the evaluation in CI would require an Anthropic API key stored as a repository secret, and that job isn't configured in this repository.

## Design decisions

- **Semantic evaluation instead of exact string matching.** The judge checks whether the answer is correct in meaning, so differences in wording don't cause false failures.
- **LLM-as-judge.** A language model is used to compare meaning, which exact matching can't do.
- **Structured judge result.** The judge returns a `passed` boolean and a `reason`. Pydantic validates both, so malformed verdicts are rejected rather than silently accepted.
- **JSON result files instead of a database.** Each run is a plain file that can be diffed, committed as an example, or read by other tools.
- **CLI instead of a frontend.** The tool is meant to run in a terminal and in CI.
- **No unnecessary frameworks.** The project depends only on the Anthropic SDK, `python-dotenv`, `pydantic`, and `pytest`. It doesn't use LangChain, LangGraph, a vector database, or RAG.

## Limitations and future improvements

- The tool evaluates one application interface, `generate_response(question) -> str`. Other applications need an adapter that implements it.
- The judge's verdicts depend on the model and on the expected answers. A judge can be wrong, and the reasons it gives should be read, not trusted blindly.
- The regression comparison matches tests by ID and reports PASS → FAIL changes. It doesn't detect every kind of regression, such as a change in the reason text for a test that still passes.
- Results are stored as files. There is no dashboard or history view.
- `regression.py` doesn't yet set an exit code, so it can't fail a CI job on its own.
- There is no CI job that runs the test suite or the evaluation.

## Portfolio context

This project demonstrates evaluation and regression testing for a GenAI application: defining expected behaviour, judging responses semantically, saving evidence, and detecting when a behaviour that used to pass no longer does. It is a small, readable implementation of those ideas. It is not a production evaluation platform, and it doesn't claim to catch every regression.
