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

question = input("Ask a question: ").strip()

if not question:
    raise SystemExit("Please enter a question.")

question_embedding = model.encode(
    question,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

scores = embeddings @ question_embedding
top_indices = np.argsort(scores)[::-1][:3]

for rank, index in enumerate(top_indices, start=1):
    record = records[index]

    print(f"\n--- Result {rank} ---")
    print(f"Similarity: {scores[index]:.3f}")
    print(f"Chunk: {record['chunk_id']}")
    print(f"Text: {record['text']}")
    print(f"Source: {record['source_url']}")