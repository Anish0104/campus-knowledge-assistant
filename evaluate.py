import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from generate import answer_question


PROJECT_ROOT = Path(__file__).resolve().parent

# Independent expectation, rather than importing the application's value.
EXPECTED_FALLBACK = (
    "I don't have enough information in the collected documents."
)

CASES = [
    {
        "id": "thesis_approval",
        "question": "Who approves my thesis?",
        "campus": "New Brunswick",
        "program": "MS Computer Science",
        "expected_document": "ms_cs_requirements",
        "required_phrases": [
            "The thesis must be approved by the student's thesis committee."
        ],
    },
    {
        "id": "library_eligibility",
        "question": "Who can use Rutgers interlibrary loan?",
        "campus": None,
        "program": None,
        "expected_document": "library_interlibrary_loan",
        "required_phrases": [
            "available to current Rutgers students, faculty, and staff",
            "faculty emeriti",
            "retirees",
            "visiting scholars and students",
            "RBHS volunteer faculty",
            "Rowan and CCC affiliates",
        ],
    },
    {
        "id": "textbook_restriction",
        "question": (
            "Can I borrow required textbooks through interlibrary loan?"
        ),
        "campus": None,
        "program": None,
        "expected_document": "library_interlibrary_loan",
        "required_phrases": [
            "can't facilitate interlibrary loan requests for textbooks"
        ],
    },
    {
        "id": "unsupported_dining_hours",
        "question": "What time does Busch dining hall close today?",
        "campus": "New Brunswick",
        "program": None,
        "expected_document": None,
        "required_phrases": [],
    },
    {
        "id": "unsupported_mba_thesis",
        "question": "Who approves an MBA thesis at Rutgers Newark?",
        "campus": "Newark",
        "program": "MBA",
        "expected_document": None,
        "required_phrases": [],
    },
    {
        "id": "university_wide_library_access",
        "question": "Who can use Rutgers interlibrary loan?",
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "library_interlibrary_loan",
        "required_phrases": [
            "available to current Rutgers students, faculty, and staff"
        ],
    },
]


def normalize(text: str) -> str:
    """Ignore whitespace and case for expected-content checks."""
    return " ".join(text.split()).casefold()


def check_result(case: dict, result: dict) -> list[str]:
    """Return failure explanations. An empty list means pass."""
    failures = []

    answer = result.get("answer")
    claims = result.get("claims")
    sources = result.get("sources")

    if not isinstance(answer, str):
        return ["Response is missing a string answer."]

    if not isinstance(claims, list) or not isinstance(sources, list):
        return ["Response must contain claims and sources lists."]

    if result.get("question") != case["question"]:
        failures.append("Returned question differs from the input.")

    # Unsupported questions must abstain without citing unrelated sources.
    if case["expected_document"] is None:
        if answer.strip() != EXPECTED_FALLBACK:
            failures.append("Expected the insufficient-information response.")

        if claims:
            failures.append("An unsupported question returned claims.")

        if sources:
            failures.append("An unsupported question returned sources.")

        return failures

    # Supported questions must return evidence.
    if not answer.strip() or answer.strip() == EXPECTED_FALLBACK:
        failures.append("Expected an answer, but the system abstained.")

    if not claims:
        failures.append("Expected at least one evidence claim.")

    if not sources:
        failures.append("Expected at least one supporting source.")

    if len(claims) > 1:
        failures.append("Current answer contract allows at most one claim.")

    sources_by_id = {}

    for source in sources:
        if not isinstance(source, dict):
            failures.append("A source is not an object.")
            continue

        source_id = source.get("chunk_id")

        if not isinstance(source_id, str) or not source_id:
            failures.append("A source has no valid chunk ID.")
            continue

        if source_id in sources_by_id:
            failures.append(f"Duplicate source ID: {source_id}")

        sources_by_id[source_id] = source

        if source.get("document_id") != case["expected_document"]:
            failures.append(
                f"Wrong document returned: {source.get('document_id')}"
            )

        # These are the scope rules used by this project.
        if case["campus"]:
            allowed_campuses = {
                normalize(case["campus"]),
                "university-wide",
            }

            if normalize(str(source.get("campus", ""))) not in allowed_campuses:
                failures.append("A source has an incompatible campus.")

        if case["program"]:
            allowed_programs = {
                normalize(case["program"]),
                "not program-specific",
            }

            if normalize(str(source.get("program", ""))) not in allowed_programs:
                failures.append("A source has an incompatible program.")

    quotes = []
    cited_ids = set()
    rendered_claims = []

    for claim in claims:
        if not isinstance(claim, dict):
            failures.append("A claim is not an object.")
            continue

        source_id = claim.get("source_id")
        quote = claim.get("evidence_quote")

        if not isinstance(source_id, str):
            failures.append("A claim has no valid source ID.")
            continue

        if not isinstance(quote, str) or not quote.strip():
            failures.append("A claim has an empty or invalid evidence quote.")
            continue

        quotes.append(quote)
        cited_ids.add(source_id)
        rendered_claims.append(f'"{quote.strip()}" [{source_id}]')

        source = sources_by_id.get(source_id)

        if source is None:
            failures.append(f"Claim cites a missing source: {source_id}")
            continue

        source_text = source.get("text")

        if not isinstance(source_text, str):
            failures.append(f"Source {source_id} has no text.")
            continue

        # Preserve case for verbatim evidence validation.
        normalized_quote = " ".join(quote.split())
        normalized_source = " ".join(source_text.split())

        if normalized_quote not in normalized_source:
            failures.append(f"Quote is absent from source {source_id}.")

    if set(sources_by_id) != cited_ids:
        failures.append("Returned sources and cited sources do not match.")

    # The current application displays extracted quotes, not paraphrases.
    expected_rendering = " ".join(rendered_claims)

    if answer.strip() != expected_rendering:
        failures.append("Displayed answer differs from the cited quotes.")

    # Check the selected quote itself, not the surrounding source passage.
    combined_quotes = normalize(" ".join(quotes))

    for phrase in case["required_phrases"]:
        if normalize(phrase) not in combined_quotes:
            failures.append(f"Missing expected evidence: {phrase}")

    return failures


def main() -> int:
    print("Running six development regression checks.")
    print("Ollama must be running. FastAPI does not need to be running.\n")

    records = []

    for case in CASES:
        print(f"Checking: {case['id']}", flush=True)
        started = perf_counter()

        try:
            result = answer_question(
                case["question"],
                campus=case["campus"],
                program=case["program"],
            )

            failures = check_result(case, result)
            status = "FAIL" if failures else "PASS"

            record = {
                "id": case["id"],
                "status": status,
                "seconds": round(perf_counter() - started, 3),
                "input": {
                    "question": case["question"],
                    "campus": case["campus"],
                    "program": case["program"],
                },
                "failures": failures,
                "result": result,
            }

        except Exception as error:
            # Keep running so one failure does not hide the other results.
            record = {
                "id": case["id"],
                "status": "ERROR",
                "seconds": round(perf_counter() - started, 3),
                "input": {
                    "question": case["question"],
                    "campus": case["campus"],
                    "program": case["program"],
                },
                "failures": [f"{type(error).__name__}: {error}"],
                "result": None,
            }

        records.append(record)

        print(
            f"{record['status']}: {record['id']} "
            f"({record['seconds']:.2f}s)"
        )

        if record["result"]:
            print("Answer:", record["result"]["answer"])

        for failure in record["failures"]:
            print("  Reason:", failure)

        print()

    passed = sum(record["status"] == "PASS" for record in records)
    failed = sum(record["status"] == "FAIL" for record in records)
    errors = sum(record["status"] == "ERROR" for record in records)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "development_regression_v1",
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": failed,
            "errors": errors,
        },
        "limitations": [
            "Known development cases, not a held-out accuracy benchmark.",
            "Expected phrases check selected evidence, not general semantics.",
            "Calls Python functions directly; does not test HTTP or frontend.",
            "Timings include cold model loading and are not latency benchmarks.",
        ],
        "cases": records,
    }

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Passed: {passed}/{len(records)}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print("\nThis is a regression-check result, not overall accuracy.")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())