import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data/evaluation/benchmark_v1_1.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.json"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings.npy"
OUTPUT_DIR = ROOT / "data/processed/benchmarks"

EXPECTED_FALLBACK = (
    "I don't have enough information in the collected documents."
)

CANDIDATE_K = 7
MAX_CLAIMS = 3
VALIDATOR_VERSION = "multi_claim_v1"


def normalize(text):
    return " ".join(text.split())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_digest(chunks):
    serialized = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def eligible(source, case):
    for field, wildcard in (
        ("campus", "university-wide"),
        ("program", "not program-specific"),
    ):
        requested = normalize(case.get(field) or "").casefold()
        actual = normalize(source.get(field) or "").casefold()

        if requested and actual not in {requested, wildcard}:
            return False

    return True


def ranking_metrics(ids, gold):
    first = next(
        (rank for rank, item in enumerate(ids, 1) if item in gold),
        None,
    )
    return {
        "hit_at_1": int(bool(set(ids[:1]) & gold)),
        "hit_at_5": int(bool(set(ids[:5]) & gold)),
        "recall_at_5": len(set(ids[:5]) & gold) / len(gold),
        "mrr_at_7": 1 / first if first else 0.0,
    }


def validate_response(result, case, canonical):
    """Validate response structure and check each source quotation."""
    if not isinstance(result, dict):
        raise ValueError("Response must be an object.")

    answer = result.get("answer")
    claims = result.get("claims")
    sources = result.get("sources")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Missing answer.")

    if not isinstance(claims, list) or not isinstance(sources, list):
        raise ValueError("Missing claims or sources lists.")

    if result.get("question") != case["question"]:
        raise ValueError("Returned question differs from input.")

    if answer.strip() == EXPECTED_FALLBACK:
        if claims or sources:
            raise ValueError("Abstention contains claims or sources.")
        return True, []

    if not 1 <= len(claims) <= MAX_CLAIMS:
        raise ValueError(
            f"Non-abstention must contain 1 to {MAX_CLAIMS} claims."
        )

    returned_ids = []

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Invalid source object.")

        source_id = source.get("chunk_id")
        if not isinstance(source_id, str):
            raise ValueError("Invalid source ID.")

        original = canonical.get(source_id)
        if original is None:
            raise ValueError("Source absent from saved corpus.")

        for field in (
            "text",
            "document_id",
            "campus",
            "program",
            "source_url",
        ):
            if source.get(field) != original.get(field):
                raise ValueError(
                    f"Source differs from saved corpus: {field}"
                )

        returned_ids.append(source_id)

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Duplicate returned sources.")

    checks = []
    cited_ids = set()
    seen_claims = set()
    rendered = []

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Invalid claim.")

        source_id = claim.get("source_id")
        quote = claim.get("evidence_quote")

        if not isinstance(source_id, str):
            raise ValueError("Invalid source ID.")

        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Invalid quote.")

        claim_key = (source_id, normalize(quote))
        if claim_key in seen_claims:
            raise ValueError("Duplicate evidence claim.")
        seen_claims.add(claim_key)

        original = canonical.get(source_id)

        # Preserve the original whitespace-normalized citation metric.
        # A real source quote is not necessarily a relevant answer.
        valid = bool(
            original
            and eligible(original, case)
            and normalize(quote) in normalize(original["text"])
            and source_id in returned_ids
        )

        checks.append(valid)
        cited_ids.add(source_id)
        rendered.append(f'"{quote.strip()}" [{source_id}]')

    if cited_ids != set(returned_ids):
        raise ValueError("Returned sources differ from citations.")

    # Accept paragraph separators or spaces, but not extra answer text,
    # altered quotations, missing citations, or reordered claims.
    if normalize(answer) != normalize("\n\n".join(rendered)):
        raise ValueError("Displayed answer differs from claims.")

    return False, checks


def quotes_gold_evidence(result, case):
    """
    Legacy metric: at least one quote contains a reviewed evidence phrase.

    This does NOT establish completeness or relevance of every claim.
    """
    if not isinstance(result, dict):
        return False

    phrases = [
        normalize(phrase)
        for phrase in case.get("evidence_phrases", [])
    ]
    claims = result.get("claims") or []

    return any(
        phrase in normalize(claim.get("evidence_quote", ""))
        for claim in claims
        if isinstance(claim, dict)
        and isinstance(claim.get("evidence_quote", ""), str)
        for phrase in phrases
    )


def valid_evidence_match(record):
    """Evidence match on a completed, cited, non-declined response."""
    return bool(
        not record["answer_error"]
        and not record["abstained"]
        and record["citation_checks"]
        and all(record["citation_checks"])
        and quotes_gold_evidence(record["result"], record["case"])
    )


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": (
            round(100 * numerator / denominator, 2)
            if denominator else None
        ),
    }


def latency_summary(values):
    if not values:
        return {
            "n": 0,
            "median_seconds": None,
            "p95_seconds": None,
        }

    ordered = sorted(values)
    return {
        "n": len(values),
        "median_seconds": round(median(values), 4),
        "p95_seconds": round(
            ordered[math.ceil(0.95 * len(ordered)) - 1], 4
        ),
    }


def load_inputs():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if dataset.get("labels_reviewed") is not True:
        raise ValueError("Review benchmark labels before running.")

    expected_hash = dataset.get("label_corpus_sha256")
    if not expected_hash:
        raise ValueError("Missing label_corpus_sha256.")

    if corpus_digest(chunks) != expected_hash:
        raise ValueError(
            "Corpus differs from reviewed labels. "
            "Review the labels before benchmarking."
        )

    canonical = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(canonical) != len(chunks):
        raise ValueError("Duplicate corpus chunk IDs.")

    cases = dataset["cases"]
    if not cases:
        raise ValueError("Benchmark contains no cases.")

    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs.")

    for case in cases:
        if not isinstance(case["answerable"], bool):
            raise ValueError(
                f"answerable must be boolean: {case['id']}"
            )

        gold = case["relevant_chunk_ids"]
        if bool(gold) != case["answerable"]:
            raise ValueError(f"Inconsistent labels: {case['id']}")

        if len(gold) != len(set(gold)):
            raise ValueError(f"Duplicate gold labels: {case['id']}")

        for chunk_id in gold:
            if chunk_id not in canonical:
                raise ValueError(f"Unknown gold chunk: {chunk_id}")
            if not eligible(canonical[chunk_id], case):
                raise ValueError(f"Gold chunk outside scope: {case['id']}")

    if not any(case["answerable"] for case in cases):
        raise ValueError("At least one supported case is required.")

    return dataset, canonical


def summarize(records):
    supported = [r for r in records if r["case"]["answerable"]]
    unsupported = [r for r in records if not r["case"]["answerable"]]
    completed = [r for r in records if not r["answer_error"]]

    checks = [
        check
        for record in completed
        for check in record["citation_checks"]
    ]

    retrieval = {
        method: {
            metric: round(
                mean(record[method][metric] for record in supported),
                4,
            )
            for metric in (
                "hit_at_1", "hit_at_5", "recall_at_5", "mrr_at_7"
            )
        }
        for method in ("embedding", "reranked")
    }

    return {
        "retrieval": retrieval,
        "unsupported_abstention": ratio(
            sum(
                bool(r["abstained"]) and not r["answer_error"]
                for r in unsupported
            ),
            len(unsupported),
        ),
        "supported_false_abstention": ratio(
            sum(
                bool(r["abstained"]) and not r["answer_error"]
                for r in supported
            ),
            len(supported),
        ),
        "answer_errors": ratio(
            sum(bool(r["answer_error"]) for r in records),
            len(records),
        ),
        "retrieval_errors": sum(
            bool(r["retrieval_error"]) for r in records
        ),
        "verbatim_citation_validity_on_completed_responses": ratio(
            sum(checks), len(checks)
        ),
        # Retained exactly as a legacy calculation for comparison.
        "supported_quotes_gold_evidence": ratio(
            sum(
                quotes_gold_evidence(r["result"], r["case"])
                for r in supported
            ),
            len(supported),
        ),
        "supported_valid_responses_with_gold_evidence": ratio(
            sum(valid_evidence_match(r) for r in supported),
            len(supported),
        ),
        "multi_claim_completed_responses": sum(
            len((r["result"] or {}).get("claims", [])) > 1
            for r in completed
        ),
        "supported_answer_correctness": "PENDING_MANUAL_REVIEW",
        "successful_call_latency": latency_summary([
            r["answer_seconds"] for r in completed
        ]),
        "successful_answerable_call_latency": latency_summary([
            r["answer_seconds"]
            for r in supported if not r["answer_error"]
        ]),
        "successful_unsupported_call_latency": latency_summary([
            r["answer_seconds"]
            for r in unsupported if not r["answer_error"]
        ]),
    }


def write_outputs(run_dir, records, report):
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = [
        "id",
        "question",
        "answerable",
        "expected_answer",
        "answer",
        "model_needed",
        "model_answer_words",
        "declined_reason",
        "quotes_gold_evidence",
        "error",
        "correct",
        "notes",
        "claim_count",
        "claims_json",
        "additional_evidence_json",
        "citation_checks_json",
        "valid_response_with_gold_evidence",
    ]

    with (run_dir / "review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for record in records:
            case = record["case"]
            result = record["result"]
            if not isinstance(result, dict):
                result = {}

            selection = result.get("selection") or {}
            claims = result.get("claims") or []

            writer.writerow({
                "id": case["id"],
                "question": case["question"],
                "answerable": case["answerable"],
                "expected_answer": case["expected_answer"],
                "answer": result.get("answer", ""),
                "model_needed": selection.get("needed", ""),
                "model_answer_words": selection.get("answer_words", ""),
                "declined_reason": selection.get("declined_reason", ""),
                "quotes_gold_evidence": (
                    quotes_gold_evidence(result, case)
                    if case["answerable"] else ""
                ),
                "error": record["answer_error"] or "",
                "correct": "",
                "notes": "",
                "claim_count": len(claims),
                "claims_json": json.dumps(claims, ensure_ascii=False),
                "additional_evidence_json": json.dumps(
                    selection.get("additional_evidence", []),
                    ensure_ascii=False,
                ),
                "citation_checks_json": json.dumps(
                    record["citation_checks"]
                ),
                "valid_response_with_gold_evidence": (
                    valid_evidence_match(record)
                    if case["answerable"] else ""
                ),
            })

    report["summary"] = summarize(records)
    report["validator_version"] = VALIDATOR_VERSION
    report["metric_notes"] = {
        "supported_quotes_gold_evidence": (
            "Legacy metric: any returned quote matches any reviewed "
            "phrase, even if another response-contract check fails."
        ),
        "supported_valid_responses_with_gold_evidence": (
            "Requires no answer error, no decline, valid citations, "
            "and at least one reviewed phrase match."
        ),
        "correctness": (
            "Neither evidence-match metric proves completeness or "
            "relevance of all claims. Review the full answer."
        ),
        "latency": (
            "Sequential Python-call timings, not HTTP or load-test "
            "measurements. Rescoring reuses original call timings."
        ),
    }

    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nBENCHMARK SUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved run: {run_dir}")
    print(f"Review file: {run_dir / 'review.csv'}")
    print("Answer correctness remains pending manual review.")


def new_run_dir(prefix=""):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = OUTPUT_DIR / f"{prefix}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def rescore_saved_run(source_dir):
    """Revalidate stored responses without loading or calling models."""
    source_dir = source_dir.resolve()
    dataset, canonical = load_inputs()

    old_report = json.loads(
        (source_dir / "report.json").read_text(encoding="utf-8")
    )

    old_hashes = old_report.get("hashes", {})
    if old_hashes.get("dataset") != digest(DATASET_PATH):
        raise ValueError("Dataset differs from the saved run.")

    if old_hashes.get("chunks") != digest(CHUNKS_PATH):
        raise ValueError("Chunk corpus differs from the saved run.")

    records = [
        json.loads(line)
        for line in (source_dir / "cases.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected_cases = {case["id"]: case for case in dataset["cases"]}
    if (
        len(records) != len(expected_cases)
        or {r["id"] for r in records} != set(expected_cases)
    ):
        raise ValueError("Saved run has missing or duplicate cases.")

    for record in records:
        if record["case"] != expected_cases[record["id"]]:
            raise ValueError(f"Saved case changed: {record['id']}")

        previous_error = record.get("answer_error")
        record["previous_answer_error"] = previous_error
        record["abstained"] = False
        record["citation_checks"] = []

        if record.get("result") is None:
            # A generation failure produced no response to revalidate.
            # Keep its original error; never turn it into a decline.
            record["answer_error"] = (
                previous_error or "No saved application response."
            )
            continue

        try:
            abstained, checks = validate_response(
                record["result"], record["case"], canonical
            )
            record["abstained"] = abstained
            record["citation_checks"] = checks
            record["answer_error"] = None
        except Exception as error:
            record["answer_error"] = (
                f"{type(error).__name__}: {error}"
            )

        if previous_error != record["answer_error"]:
            print(
                f"{record['id']}: "
                f"{previous_error or 'no error'} -> "
                f"{record['answer_error'] or 'no error'}"
            )

    report = dict(old_report)
    report["original_created_at"] = old_report.get("created_at")
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["run_type"] = "rescore"
    report["rescored_from"] = str(source_dir)
    report["rescore_benchmark_sha256"] = digest(Path(__file__))
    report["rescore_git_commit"] = git_output("rev-parse", "HEAD")
    report["rescore_git_status"] = git_output("status", "--short")
    report["rescore_note"] = (
        "No model calls were made. Original generation hashes, settings, "
        "environment, and call timings are preserved. Manual review "
        "columns are blank; original files remain untouched."
    )

    run_dir = new_run_dir("rescore_")
    write_outputs(run_dir, records, report)


def run_benchmark():
    dataset, canonical = load_inputs()
    cases = dataset["cases"]

    tracked_paths = {
        "dataset": DATASET_PATH,
        "chunks": CHUNKS_PATH,
        "embeddings": EMBEDDINGS_PATH,
        "search.py": ROOT / "search.py",
        "rerank.py": ROOT / "rerank.py",
        "generate.py": ROOT / "generate.py",
        "config.py": ROOT / "config.py",
        "benchmark.py": Path(__file__).resolve(),
    }
    initial_hashes = {
        name: digest(path) for name, path in tracked_paths.items()
    }

    from config import OLLAMA_MODEL
    from search import search
    from rerank import get_reranker
    from generate import (
        answer_question,
        MIN_RERANK_SCORE,
        GENERATION_TOP_K,
        CANDIDATE_K as GENERATION_CANDIDATE_K,
        FALLBACK_ANSWER,
    )

    if GENERATION_CANDIDATE_K != CANDIDATE_K:
        raise ValueError("Generation and benchmark candidate counts differ.")

    if not 1 <= GENERATION_TOP_K <= CANDIDATE_K:
        raise ValueError("Require 1 <= GENERATION_TOP_K <= CANDIDATE_K.")

    if FALLBACK_ANSWER != EXPECTED_FALLBACK:
        raise ValueError("Generation and benchmark fallbacks differ.")

    settings = {
        "candidate_k": CANDIDATE_K,
        "generation_top_k": GENERATION_TOP_K,
        "min_rerank_score": MIN_RERANK_SCORE,
        "ollama_model": OLLAMA_MODEL,
        "max_claims": MAX_CLAIMS,
    }

    print("BENCHMARK SETTINGS")
    print(json.dumps(settings, indent=2))

    warmup_started = perf_counter()
    get_reranker()

    # Preserve fail-fast behavior if warm-up cannot generate a response.
    answer_question(
        "Who approves my thesis?",
        campus="New Brunswick",
        program="MS Computer Science",
    )
    settings["warmup_seconds"] = round(
        perf_counter() - warmup_started, 4
    )

    run_dir = new_run_dir()
    print(f"\nRun folder: {run_dir}")
    records = []

    for case in cases:
        print(
            f"\nChecking {case['id']}: {case['question']}",
            flush=True,
        )
        record = {
            "id": case["id"],
            "case": case,
            "retrieval_error": None,
            "answer_error": None,
            "result": None,
            "abstained": False,
            "citation_checks": [],
        }
        scope = {
            "campus": case["campus"],
            "program": case["program"],
        }

        if case["answerable"]:
            gold = set(case["relevant_chunk_ids"])
            try:
                candidates = search(
                    case["question"],
                    top_k=CANDIDATE_K,
                    **scope,
                )
                original_ids = [p["chunk_id"] for p in candidates]

                if candidates:
                    scores = get_reranker().predict([
                        (case["question"], p["text"])
                        for p in candidates
                    ])
                    ranked = sorted(
                        zip(candidates, scores),
                        key=lambda pair: float(pair[1]),
                        reverse=True,
                    )
                    reranked_ids = [p["chunk_id"] for p, _ in ranked]
                else:
                    reranked_ids = []

                record["embedding"] = ranking_metrics(original_ids, gold)
                record["reranked"] = ranking_metrics(reranked_ids, gold)
                record["embedding_ids"] = original_ids
                record["reranked_ids"] = reranked_ids

            except Exception as error:
                record["retrieval_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                record["embedding"] = ranking_metrics([], gold)
                record["reranked"] = ranking_metrics([], gold)

        started = perf_counter()
        try:
            result = answer_question(case["question"], **scope)
            record["result"] = result

            abstained, checks = validate_response(
                result, case, canonical
            )
            record["abstained"] = abstained
            record["citation_checks"] = checks

        except Exception as error:
            record["answer_error"] = (
                f"{type(error).__name__}: {error}"
            )
            print(f"Answer error: {record['answer_error']}", flush=True)

        record["answer_seconds"] = perf_counter() - started
        records.append(record)

        # Save each response immediately in case a later call is interrupted.
        with (run_dir / "cases.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for name, path in tracked_paths.items():
        if digest(path) != initial_hashes[name]:
            raise ValueError(
                f"{name} changed during benchmark. Discard this run."
            )

    report = {
        "name": dataset["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": "live",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "hashes": initial_hashes,
        "label_corpus_sha256": dataset["label_corpus_sha256"],
        "label_revision": dataset.get("label_revision"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "settings": settings,
        "limitations": [
            "Curated source-derived questions on five documents.",
            "Not an independent general-accuracy benchmark.",
            "Citation matching normalizes whitespace.",
            "Citation validity does not establish answer relevance.",
            "Evidence matches do not establish full-answer correctness.",
            "Correctness requires manual review of all returned claims.",
            "Answer errors are separate from successful abstentions.",
            "Corpus hashing does not prove embeddings match the corpus.",
        ],
    }
    write_outputs(run_dir, records, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rescore",
        type=Path,
        help="Revalidate an existing run without making model calls.",
    )
    args = parser.parse_args()

    if args.rescore:
        rescore_saved_run(args.rescore)
    else:
        run_benchmark()


if __name__ == "__main__":
    main()