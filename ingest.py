import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "data" / "documents.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 150
OVERLAP = 30


def chunk_text(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_SIZE - OVERLAP
    chunks = []

    for start in range(0, len(words), step):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

    return chunks


def main() -> None:
    documents = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(documents, list) or not documents:
        raise ValueError("The document catalog must be a non-empty list.")

    records = []
    seen_ids = set()

    required_fields = {
        "document_id",
        "filename",
        "source_url",
        "campus",
        "program",
        "topic",
        "effective_year",
    }

    for document in documents:
        missing = required_fields - document.keys()

        if missing:
            raise ValueError(
                f"Catalog entry is missing fields: {sorted(missing)}"
            )

        document_id = document["document_id"]

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Each document needs a non-empty document_id.")

        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id: {document_id}")

        seen_ids.add(document_id)

        document_path = RAW_DIR / document["filename"]
        text = document_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"Document is empty: {document_path.name}")

        chunks = chunk_text(text)

        metadata = {
            key: value
            for key, value in document.items()
            if key != "filename"
        }

        for index, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    **metadata,
                    "chunk_id": f"{document_id}_{index:03d}",
                    "text": chunk,
                }
            )

        print(
            f"{document_id}: "
            f"{len(text.split())} words -> {len(chunks)} chunks"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDocuments processed: {len(documents)}")
    print(f"Total chunks: {len(records)}")
    print(f"Saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()