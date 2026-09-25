import json
from functools import lru_cache

import numpy as np

from config import CHUNKS_PATH, EMBEDDING_MODEL, EMBEDDINGS_PATH


@lru_cache(maxsize=1)
def load_index() -> tuple[list[dict], np.ndarray]:
    """Load the saved chunks and embeddings once per Python process."""
    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(records) != len(embeddings):
        raise ValueError(
            f"{len(records)} chunks but {len(embeddings)} embeddings. "
            "Rerun ingest.py and embed.py."
        )

    return records, embeddings


@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once per Python process."""
    # Imported here so tests and tools that never search do not
    # need to load PyTorch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def normalize(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def in_scope(
    record: dict,
    campus: str | None,
    program: str | None,
) -> bool:
    """
    Keep a chunk when it matches the requested campus and program.

    University-wide and non-program-specific chunks match any request.
    """
    requested_campus = normalize(campus)
    requested_program = normalize(program)

    record_campus = normalize(record.get("campus"))
    record_program = normalize(record.get("program"))

    campus_matches = (
        not requested_campus
        or record_campus in {requested_campus, "university-wide"}
    )

    program_matches = (
        not requested_program
        or record_program in {requested_program, "not program-specific"}
    )

    return campus_matches and program_matches


def search(
    question: str,
    top_k: int = 3,
    *,
    campus: str | None = None,
    program: str | None = None,
) -> list[dict]:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    records, embeddings = load_index()

    eligible_indices = [
        index
        for index, record in enumerate(records)
        if in_scope(record, campus, program)
    ]

    if not eligible_indices:
        return []

    question_embedding = get_embedder().encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    eligible_embeddings = embeddings[eligible_indices]
    scores = eligible_embeddings @ question_embedding
    ranked_positions = np.argsort(scores)[::-1][:top_k]

    return [
        {
            **records[eligible_indices[position]],
            "similarity": float(scores[position]),
        }
        for position in ranked_positions
    ]


if __name__ == "__main__":
    question = input("Ask a question: ")

    try:
        results = search(question)

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Text: {result['text']}")
            print(f"Source: {result['source_url']}")

    except ValueError as error:
        print(f"Invalid input: {error}")
