"""
Recompute benchmark gold chunk labels from reviewed evidence phrases.

Each answerable case lists `evidence_phrases`: exact text from the
source documents that is sufficient to answer the question. A chunk is
gold when it contains every phrase for the case and is in the case's
campus/program scope. Because phrases are tied to the documents rather
than to chunk IDs, labels can be recomputed after chunking changes.

Usage:
    python data/evaluation/relabel.py            # show changes only
    python data/evaluation/relabel.py --write    # back up and save

Review the printed changes before using --write. A phrase that is
missing from every chunk is an error, never silently dropped.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data/evaluation/benchmark_v1_1.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.json"

sys.path.insert(0, str(ROOT))

from search import in_scope  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(text.split())


def corpus_digest(chunks: list[dict]) -> str:
    """Same canonical hash that benchmark.py checks."""
    serialized = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def gold_chunks(case: dict, chunks: list[dict]) -> list[str]:
    phrases = [normalize(p) for p in case["evidence_phrases"]]

    return [
        chunk["chunk_id"]
        for chunk in chunks
        if in_scope(chunk, case["campus"], case["program"])
        and all(p in normalize(chunk["text"]) for p in phrases)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    problems = []
    changes = []

    for case in dataset["cases"]:
        if not case["answerable"]:
            if case.get("evidence_phrases"):
                problems.append(f"{case['id']}: unanswerable case has phrases")
            continue

        if not case.get("evidence_phrases"):
            problems.append(f"{case['id']}: missing evidence_phrases")
            continue

        new = gold_chunks(case, chunks)

        if not new:
            problems.append(
                f"{case['id']}: no in-scope chunk contains all phrases"
            )
            continue

        if new != case["relevant_chunk_ids"]:
            changes.append((case, new))

    for problem in problems:
        print("ERROR", problem)

    for case, new in changes:
        print(f"{case['id']}: {case['relevant_chunk_ids']} -> {new}")
        print(f"  {case['question']}")

    new_hash = corpus_digest(chunks)
    hash_changed = new_hash != dataset.get("label_corpus_sha256")

    print(f"\nLabel changes: {len(changes)}")
    print(f"Corpus hash changed: {hash_changed}")

    if problems:
        print("Fix the errors above before writing.")
        return 1

    if not args.write:
        print("Dry run. Re-run with --write to save.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}.before_relabel_{stamp}.json"
    )
    shutil.copy2(DATASET_PATH, backup)

    for case, new in changes:
        case["relevant_chunk_ids"] = new

    dataset["label_corpus_sha256"] = new_hash

    if args.revision:
        dataset["label_revision"] = args.revision

    DATASET_PATH.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved. Backup: {backup.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
