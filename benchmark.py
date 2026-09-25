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

# Keep retrieval evaluation fixed at seven candidates.
CANDIDATE_K = 7


def normalize(text):
    return " ".join(text.split())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_digest(chunks):
    """Match the canonical JSON hash used during label review."""
    serialized = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


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
        requested = normalize(
            case.get(field) or ""
        ).casefold()
        actual = normalize(
            source.get(field) or ""
        ).casefold()

        if requested and actual not in {requested, wildcard}:
            return False

    return True


def ranking_metrics(ids, gold):
    first = next(
        (
            rank
            for rank, item in enumerate(ids, 1)
            if item in gold
        ),
        None,
    )

    return {
        "hit_at_1": int(bool(set(ids[:1]) & gold)),
        "hit_at_5": int(bool(set(ids[:5]) & gold)),
        "recall_at_5": len(set(ids[:5]) & gold) / len(gold),
        "mrr_at_7": 1 / first if first else 0.0,
    }


def validate_response(result, case, canonical):
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

    if len(claims) > 1:
        raise ValueError(
            "Application contract permits at most one claim."
        )

    if answer.strip() == EXPECTED_FALLBACK:
        if claims or sources:
            raise ValueError(
                "Abstention contains claims or sources."
            )
        return True, []

    if not claims:
        raise ValueError(
            "Non-abstention has no evidence claim."
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
            raise ValueError(
                "Source absent from saved corpus."
            )

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

        original = canonical.get(source_id)

        # Preserves the previous benchmark's whitespace-normalized
        # citation check. This does not establish answer correctness.
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
        raise ValueError(
            "Returned sources differ from citations."
        )

    if answer.strip() != " ".join(rendered):
        raise ValueError(
            "Displayed answer differs from claims."
        )

    return False, checks


def quotes_gold_evidence(result, case):
    """
    True when a returned quote contains one of the case's reviewed
    evidence phrases. A valid citation can still quote the wrong
    sentence; this checks that it quotes the right one.
    """
    if not isinstance(result, dict):
        return False

    phrases = [normalize(p) for p in case.get("evidence_phrases", [])]

    return any(
        phrase in normalize(claim.get("evidence_quote", ""))
        for claim in result.get("claims") or []
        for phrase in phrases
    )


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": (
            round(100 * numerator / denominator, 2)
            if denominator
            else None
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
            ordered[math.ceil(0.95 * len(ordered)) - 1],
            4,
        ),
    }


def main():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    if dataset.get("labels_reviewed") is not True:
        raise ValueError(
            "Review benchmark labels before running."
        )

    cases = dataset["cases"]
    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    expected_corpus_hash = dataset.get("label_corpus_sha256")

    if not expected_corpus_hash:
        raise ValueError(
            "Missing label_corpus_sha256. "
            "Apply the reviewed label update before benchmarking."
        )

    if corpus_digest(chunks) != expected_corpus_hash:
        raise ValueError(
            "Corpus differs from the corpus used for label review. "
            "Review the chunk labels before benchmarking."
        )

    canonical = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    if len(canonical) != len(chunks):
        raise ValueError("Duplicate corpus chunk IDs.")

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
            raise ValueError(
                f"Inconsistent labels: {case['id']}"
            )

        if len(gold) != len(set(gold)):
            raise ValueError(
                f"Duplicate gold labels: {case['id']}"
            )

        for chunk_id in gold:
            if chunk_id not in canonical:
                raise ValueError(
                    f"Unknown gold chunk: {chunk_id}"
                )

            if not eligible(canonical[chunk_id], case):
                raise ValueError(
                    f"Gold chunk outside scope: {case['id']}"
                )

    if not any(case["answerable"] for case in cases):
        raise ValueError(
            "At least one supported case is required."
        )

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
        name: digest(path)
        for name, path in tracked_paths.items()
    }

    # Imports load the embedding model.
    # Model loading is excluded from request timings.
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
        raise ValueError(
            "Generation and benchmark candidate counts differ. "
            f"Expected {CANDIDATE_K}, "
            f"got {GENERATION_CANDIDATE_K}."
        )

    if not 1 <= GENERATION_TOP_K <= CANDIDATE_K:
        raise ValueError(
            "Require 1 <= GENERATION_TOP_K <= CANDIDATE_K."
        )

    if FALLBACK_ANSWER != EXPECTED_FALLBACK:
        raise ValueError(
            "Generation fallback differs from benchmark fallback."
        )

    settings = {
        "candidate_k": CANDIDATE_K,
        "generation_top_k": GENERATION_TOP_K,
        "min_rerank_score": MIN_RERANK_SCORE,
        "ollama_model": OLLAMA_MODEL,
    }

    print("BENCHMARK SETTINGS")
    print(json.dumps(settings, indent=2))

    # Use an existing development question for warm-up.
    warmup_started = perf_counter()
    get_reranker()

    answer_question(
        "Who approves my thesis?",
        campus="New Brunswick",
        program="MS Computer Science",
    )

    warmup_seconds = perf_counter() - warmup_started
    settings["warmup_seconds"] = round(warmup_seconds, 4)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    run_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"\nRun folder: {run_dir.relative_to(ROOT)}")

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

        # Evaluate both rankings on the identical candidate pool.
        if case["answerable"]:
            try:
                candidates = search(
                    case["question"],
                    top_k=CANDIDATE_K,
                    **scope,
                )

                original_ids = [
                    passage["chunk_id"]
                    for passage in candidates
                ]

                if candidates:
                    scores = get_reranker().predict([
                        (case["question"], passage["text"])
                        for passage in candidates
                    ])

                    reranked = sorted(
                        zip(candidates, scores),
                        key=lambda pair: float(pair[1]),
                        reverse=True,
                    )

                    reranked_ids = [
                        passage["chunk_id"]
                        for passage, _ in reranked
                    ]
                else:
                    reranked_ids = []

                gold = set(case["relevant_chunk_ids"])

                record["embedding"] = ranking_metrics(
                    original_ids, gold
                )
                record["reranked"] = ranking_metrics(
                    reranked_ids, gold
                )
                record["embedding_ids"] = original_ids
                record["reranked_ids"] = reranked_ids

            except Exception as error:
                record["retrieval_error"] = (
                    f"{type(error).__name__}: {error}"
                )

                # Retrieval errors count as zero.
                gold = set(case["relevant_chunk_ids"])
                record["embedding"] = ranking_metrics([], gold)
                record["reranked"] = ranking_metrics([], gold)

        # Time the actual application call, including retrieval.
        started = perf_counter()

        try:
            result = answer_question(
                case["question"], **scope
            )
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
            print(
                f"Answer error: {record['answer_error']}",
                flush=True,
            )

        record["answer_seconds"] = perf_counter() - started
        records.append(record)

        with (run_dir / "cases.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    supported = [
        record for record in records
        if record["case"]["answerable"]
    ]
    unsupported = [
        record for record in records
        if not record["case"]["answerable"]
    ]
    completed = [
        record for record in records
        if not record["answer_error"]
    ]

    checks = [
        check
        for record in completed
        for check in record["citation_checks"]
    ]

    retrieval = {
        method: {
            metric: round(
                mean(
                    record[method][metric]
                    for record in supported
                ),
                4,
            )
            for metric in (
                "hit_at_1",
                "hit_at_5",
                "recall_at_5",
                "mrr_at_7",
            )
        }
        for method in ("embedding", "reranked")
    }

    summary = {
        "retrieval": retrieval,
        "unsupported_abstention": ratio(
            sum(
                record["abstained"]
                and not record["answer_error"]
                for record in unsupported
            ),
            len(unsupported),
        ),
        "supported_false_abstention": ratio(
            sum(
                record["abstained"]
                and not record["answer_error"]
                for record in supported
            ),
            len(supported),
        ),
        "answer_errors": ratio(
            sum(
                bool(record["answer_error"])
                for record in records
            ),
            len(records),
        ),
        "retrieval_errors": sum(
            bool(record["retrieval_error"])
            for record in records
        ),
        "verbatim_citation_validity_on_completed_responses": ratio(
            sum(checks), len(checks)
        ),
        "supported_quotes_gold_evidence": ratio(
            sum(
                quotes_gold_evidence(record["result"], record["case"])
                for record in supported
            ),
            len(supported),
        ),
        "supported_answer_correctness": "PENDING_MANUAL_REVIEW",
        "successful_call_latency": latency_summary([
            record["answer_seconds"]
            for record in completed
        ]),
        "successful_answerable_call_latency": latency_summary([
            record["answer_seconds"]
            for record in supported
            if not record["answer_error"]
        ]),
        "successful_unsupported_call_latency": latency_summary([
            record["answer_seconds"]
            for record in unsupported
            if not record["answer_error"]
        ]),
    }

    with (run_dir / "review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()

        for record in records:
            case = record["case"]
            result = record["result"]
            if not isinstance(result, dict):
                result = {}

            selection = result.get("selection") or {}

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
            })

    # Detect changes to data or code while the run was active.
    for name, path in tracked_paths.items():
        if digest(path) != initial_hashes[name]:
            raise ValueError(
                f"{name} changed during benchmark. "
                "Discard this run."
            )

    report = {
        "name": dataset["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "hashes": initial_hashes,
        "label_corpus_sha256": expected_corpus_hash,
        "label_revision": dataset.get("label_revision"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "settings": settings,
        "summary": summary,
        "limitations": [
            "Curated source-derived questions on five documents.",
            "Not an independent general-accuracy benchmark.",
            "Gold chunk labels are derived from reviewed evidence phrases; "
        "review phrase changes, not only chunk IDs.",
            "Citation validity does not establish answer relevance.",
            "Citation matching normalizes whitespace.",
            "Correctness requires manual review of complete answers.",
            "Sequential Python-call latency after warm-up, not HTTP latency.",
            "One call per case; latency is exploratory, not a load test.",
            "Answer errors are separate from successful abstentions.",
            "Corpus hashing does not prove embeddings match the corpus.",
        ],
    }

    (run_dir / "report.json").write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    print("\nBENCHMARK SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved run: {run_dir.relative_to(ROOT)}")
    print(f"Review file: {run_dir / 'review.csv'}")
    print(
        "Answer correctness remains pending "
        "until review.csv is graded."
    )


if __name__ == "__main__":
    main()