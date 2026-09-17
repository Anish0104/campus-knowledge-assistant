import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
processed_dir = PROJECT_ROOT / "data" / "processed"

chunks_path = processed_dir / "ms_cs_requirements_chunks.json"
records = json.loads(chunks_path.read_text(encoding="utf-8"))

texts = [record["text"] for record in records]

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device="cpu")

# Check that every passage fits within the model's input limit.
for record in records:
    token_ids = model.tokenizer(
        record["text"],
        truncation=False,
    )["input_ids"]

    if len(token_ids) > model.max_seq_length:
        raise ValueError(
            f"{record['chunk_id']} has {len(token_ids)} tokens; "
            f"the limit is {model.max_seq_length}. "
            "Reduce the chunk size and rerun ingest.py."
        )

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

output_path = processed_dir / "ms_cs_requirements_embeddings.npy"
np.save(output_path, embeddings)

print(f"Passages embedded: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First vector, first 5 numbers: {embeddings[0][:5]}")
print(f"Saved to: {output_path.name}")