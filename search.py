import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
processed_dir = PROJECT_ROOT / "data" / "processed"

records = json.loads(
    (processed_dir / "chunks.json").read_text(
        encoding="utf-8"
    )
)

embeddings = np.load(
    processed_dir / "embeddings.npy"
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

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

    def normalize(value: str | None) -> str:
        return " ".join((value or "").split()).casefold()

    requested_campus = normalize(campus)
    requested_program = normalize(program)

    eligible_indices = []

    for index, record in enumerate(records):
        record_campus = normalize(record.get("campus"))
        record_program = normalize(record.get("program"))

        campus_matches = (
            not requested_campus
            or record_campus == requested_campus
            or record_campus == "university-wide"
        )

        program_matches = (
            not requested_program
            or record_program == requested_program
            or record_program == "not program-specific"
        )

        if campus_matches and program_matches:
            eligible_indices.append(index)

    if not eligible_indices:
        return []

    question_embedding = model.encode(
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