from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent
document_path = PROJECT_ROOT / "data" / "raw" / "ms_cs_requirements.txt"

text = document_path.read_text(encoding="utf-8")

print(f"Document: {document_path.name}")
print(f"Characters: {len(text)}")
print(f"Words: {len(text.split())}")

print("\nPreview:\n")
print(text[:500])

words = text.split()

chunk_size = 150
overlap = 30
step = chunk_size - overlap

chunks = []

for start in range(0, len(words), step):
    end = start + chunk_size
    chunk = " ".join(words[start:end])
    chunks.append(chunk)

    if end >= len(words):
        break

print(f"\nTotal chunks: {len(chunks)}")

for number, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {number} | {len(chunk.split())} words ---")
    print(chunk)


records = []

for index, chunk in enumerate(chunks, start=1):
    record = {
        "chunk_id": f"ms_cs_requirements_{index:03d}",
        "document_id": "ms_cs_requirements",
        "text": chunk,
        "source_url": (
            "https://www.cs.rutgers.edu/academics/graduate/"
            "m-s-program/degree-requirements"
        ),
        "campus": "New Brunswick",
        "program": "MS Computer Science",
        "effective_year": None,
    }
    records.append(record)

output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "ms_cs_requirements_chunks.json"
output_path.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nSaved {len(records)} chunks to {output_path.name}")
    