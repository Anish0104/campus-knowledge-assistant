import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
processed_dir = PROJECT_ROOT / "data" / "processed"

records = json.loads(
    (processed_dir / "ms_cs_requirements_chunks.json").read_text(
        encoding="utf-8"
    )
)

embeddings = np.load(
    processed_dir / "ms_cs_requirements_embeddings.npy"
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

def search(question: str, top_k: int = 3) -> list[dict]:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        record = records[index]

        results.append({
            **record,
            "similarity": float(scores[index]),
        })

    return results


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