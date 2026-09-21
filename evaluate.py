import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from generate import answer_question


PROJECT_ROOT = Path(__file__).resolve().parent

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
        "quality_forbidden_phrases": ["Textbooks?"],
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
    {
        "id": "eduroam_definition",
        "question": "What is eduroam?",
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "it_eduroam",
        "required_phrases": [
            "free and secure wireless network",
            "Rutgers community",
            "participating universities",
        ],
        "quality_forbidden_phrases": ["eduroam eduroam"],
        "quality_max_words": 30,
    },
    {
        "id": "eduroam_visitor_support",
        "question": (
            "Who should visitors contact for help connecting to eduroam?"
        ),
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "it_eduroam",
        "required_phrases": [
            "Visitors should contact IT staff at their home institution"
        ],
        "quality_forbidden_phrases": [
            "Help & Support",
            "Get help",
        ],
        "quality_max_words": 35,
    },
    {
        "id": "duo_purpose",
        "question": "What does two-step login with Duo do?",
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "it_two_step_login",
        "required_phrases": [],
        # Accept either a security explanation or the identity check.
        # Each alternative must contain every phrase in its group.
        "required_phrase_alternatives": [
            [
                "helps protect your account",
                "extra layer of security",
                "beyond your NetID and password",
            ],
            [
                "three-digit code",
                "confirm your identity",
            ],
        ],
        "quality_preferred_phrases": [
            "extra layer of security",
        ],
        "quality_max_words": 55,
    },
    {
        "id": "office_installation_limit",
        "question": (
            "How many computers can I install Microsoft Office on?"
        ),
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "it_microsoft_office",
        "required_phrases": [
            "up to five computers",
        ],
        "quality_max_words": 60,
    },
    {
        "id": "office_license_expiration",
        "question": (
            "What happens to my Microsoft Office license "
            "when I leave Rutgers?"
        ),
        "campus": "Newark",
        "program": "MBA",
        "expected_document": "it_microsoft_office",
        "required_phrases": [
            "the license will expire",
            "30 days",
            "read-only mode with limited functionality",
        ],
        "quality_max_words": 70,
    },
    {
        "id": "unsupported_duo_token_replacement",
        "question": "How do I replace a lost Duo hardware token?",
        "campus": "Newark",
        "program": "MBA",
        "expected_document": None,
        "required_phrases": [],
    },
]


def normalize(text: str) -> str:
    """Ignore whitespace and case for expected-content checks."""
    return " ".join(text.split()).casefold()


def check_result(case: dict, result: dict) -> list[str]:
    """Check the response contract and case-specific evidence."""
    if not isinstance(result, dict):
        return ["Response is not an object."]

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

    if case["expected_document"] is None:
        if answer.strip() != EXPECTED_FALLBACK:
            failures.append("Expected the insufficient-information response.")

        if claims:
            failures.append("An unsupported question returned claims.")

        if sources:
            failures.append("An unsupported question returned sources.")

        return failures

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

        if not isinstance(source_id, str) or not source_id.strip():
            failures.append("A source has no valid chunk ID.")
            continue

        if source_id in sources_by_id:
            failures.append(f"Duplicate source ID: {source_id}")

        sources_by_id[source_id] = source

        if source.get("document_id") != case["expected_document"]:
            failures.append(
                f"Wrong document returned: {source.get('document_id')}"
            )

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

        if not isinstance(source_id, str) or not source_id.strip():
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

        # Case-sensitive comparison for verbatim evidence.
        normalized_quote = " ".join(quote.split())
        normalized_source = " ".join(source_text.split())

        if normalized_quote not in normalized_source:
            failures.append(f"Quote is absent from source {source_id}.")

    if set(sources_by_id) != cited_ids:
        failures.append("Returned sources and cited sources do not match.")

    expected_rendering = " ".join(rendered_claims)

    if answer.strip() != expected_rendering:
        failures.append("Displayed answer differs from the cited quotes.")

    combined_quotes = normalize(" ".join(quotes))

    for phrase in case["required_phrases"]:
        if normalize(phrase) not in combined_quotes:
            failures.append(f"Missing expected evidence: {phrase}")

    alternatives = case.get("required_phrase_alternatives", [])

    if alternatives and not any(
        all(normalize(phrase) in combined_quotes for phrase in group)
        for group in alternatives
    ):
        failures.append("Evidence does not match an accepted answer alternative.")

    return failures


def check_quality(case: dict, result: dict) -> list[str]:
    """Case-specific presentation heuristics, not semantic verification."""
    warnings = []

    quotes = [
        claim["evidence_quote"]
        for claim in result["claims"]
    ]
    combined = " ".join(quotes)
    normalized = normalize(combined)

    for phrase in case.get("quality_forbidden_phrases", []):
        if normalize(phrase) in normalized:
            warnings.append(f"Unnecessary heading or text: {phrase}")

    max_words = case.get("quality_max_words")
    word_count = len(combined.split())

    if max_words is not None and word_count > max_words:
        warnings.append(
            f"Quote contains {word_count} words; "
            f"the case's presentation target is at most {max_words}."
        )

    for phrase in case.get("quality_preferred_phrases", []):
        if normalize(phrase) not in normalized:
            warnings.append(
                f"A more direct answer would include: {phrase}"
            )

    return warnings


def main() -> int:
    print(f"Running {len(CASES)} development regression checks.")
    print("Ollama must be running. FastAPI does not need to be running.")
    print("Quality warnings are reported separately from failures.\n")

    records = []

    for case in CASES:
        print(f"Checking: {case['id']}", flush=True)
        started = perf_counter()

        record = {
            "id": case["id"],
            "input": {
                "question": case["question"],
                "campus": case["campus"],
                "program": case["program"],
            },
            "quality_status": "NOT_CHECKED",
            "quality_warnings": [],
            "result": None,
        }

        try:
            result = answer_question(
                case["question"],
                campus=case["campus"],
                program=case["program"],
            )

            record["result"] = result
            failures = check_result(case, result)
            record["failures"] = failures
            record["status"] = "FAIL" if failures else "PASS"

            quality_configured = any(
                key.startswith("quality_") for key in case
            )

            if (
                not failures
                and case["expected_document"] is not None
                and quality_configured
            ):
                warnings = check_quality(case, result)
                record["quality_warnings"] = warnings
                record["quality_status"] = (
                    "WARN" if warnings else "PASS"
                )

        except Exception as error:
            record["status"] = "ERROR"
            record["failures"] = [
                f"{type(error).__name__}: {error}"
            ]

        record["seconds"] = round(perf_counter() - started, 3)
        records.append(record)

        print(
            f"{record['status']}: {record['id']} "
            f"({record['seconds']:.2f}s)"
        )

        if isinstance(record["result"], dict):
            print("Answer:", record["result"].get("answer", "<missing>"))

        for failure in record["failures"]:
            print("  Reason:", failure)

        for warning in record["quality_warnings"]:
            print("  Quality warning:", warning)

        print()

    passed = sum(record["status"] == "PASS" for record in records)
    failed = sum(record["status"] == "FAIL" for record in records)
    errors = sum(record["status"] == "ERROR" for record in records)
    quality_checked = sum(
        record["quality_status"] != "NOT_CHECKED"
        for record in records
    )
    quality_warned = sum(
        record["quality_status"] == "WARN"
        for record in records
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "development_regression_v2",
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "quality_checked": quality_checked,
            "cases_with_quality_warnings": quality_warned,
        },
        "limitations": [
            "Known development cases, not a held-out accuracy benchmark.",
            "Phrase checks do not establish general semantic correctness.",
            "Quote validation compares against returned source text.",
            "Quality checks are case-specific heuristics, not a complete review.",
            "Calls Python functions directly; does not test HTTP or frontend.",
            "Timings include model loading and are not latency benchmarks.",
        ],
        "cases": records,
    }

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve each run so chunking changes can be compared later.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_path = output_dir / f"evaluation_report_{stamp}.json"
    latest_path = output_dir / "evaluation_report.json"
    serialized = json.dumps(report, indent=2, ensure_ascii=False)

    archive_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")

    print(f"Passed: {passed}/{len(records)}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(
        f"Cases with quality warnings: "
        f"{quality_warned}/{quality_checked} checked"
    )
    print(f"Report: {latest_path.relative_to(PROJECT_ROOT)}")
    print(f"Archived run: {archive_path.relative_to(PROJECT_ROOT)}")
    print("\nThis is a regression-check result, not overall accuracy.")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())