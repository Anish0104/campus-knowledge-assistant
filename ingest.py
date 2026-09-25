import json
from pathlib import Path

from text_units import join_units, text_units


PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "data" / "documents.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"

# Target chunk size, measured in words.
# An individual sentence can exceed this size to avoid cutting it.
# Sentences never cross line breaks; see text_units.py.
CHUNK_SIZE = 150

# Maximum overlap, using complete trailing sentences only.
OVERLAP = 30


def chunk_units(units: list[dict]) -> list[list[dict]]:
    """
    Pack complete units (sentences within one line) into chunks.

    Overlap contains only complete trailing units whose combined
    length fits within OVERLAP. A unit longer than CHUNK_SIZE is
    kept intact.
    """
    if not 0 <= OVERLAP < CHUNK_SIZE:
        raise ValueError("Require 0 <= OVERLAP < CHUNK_SIZE.")

    chunks = []
    current = []
    current_words = 0

    def words(unit: dict) -> int:
        return len(unit["text"].split())

    for unit in units:
        unit_words = words(unit)

        if current and current_words + unit_words > CHUNK_SIZE:
            # A heading belongs with the text after it, so trailing
            # headings are left for the next chunk.
            closed = list(current)
            while len(closed) > 1 and closed[-1].get("heading"):
                closed.pop()

            chunks.append(closed)

            overlap = []
            overlap_words = 0

            # Keep a contiguous suffix of complete units. Headings
            # removed above are always carried over.
            carried = current[len(closed):]
            carried_words = sum(words(unit) for unit in carried)

            for previous in reversed(closed):
                if carried_words + overlap_words + words(previous) > OVERLAP:
                    break

                overlap.insert(0, previous)
                overlap_words += words(previous)

            overlap += carried
            overlap_words += carried_words

            current = overlap
            current_words = overlap_words

            # Make room for the next unit by removing overlap.
            while current and current_words + unit_words > CHUNK_SIZE:
                removed = current.pop(0)
                current_words -= words(removed)

        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str) -> list[str]:
    """Chunk a document and return only the chunk texts."""
    return [join_units(chunk) for chunk in chunk_units(text_units(text))]


def main() -> None:
    documents = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(documents, list) or not documents:
        raise ValueError(
            "The document catalog must be a non-empty list."
        )

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
        if not isinstance(document, dict):
            raise ValueError(
                "Each document catalog entry must be an object."
            )

        missing = required_fields - document.keys()

        if missing:
            raise ValueError(
                f"Catalog entry is missing fields: {sorted(missing)}"
            )

        document_id = document["document_id"]

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "Each document needs a non-empty document_id."
            )

        if document_id in seen_ids:
            raise ValueError(
                f"Duplicate document_id: {document_id}"
            )

        seen_ids.add(document_id)

        filename = document["filename"]

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"Document {document_id} needs a non-empty filename."
            )

        document_path = RAW_DIR / filename
        text = document_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(
                f"Document is empty: {document_path.name}"
            )

        chunks = chunk_units(text_units(text))

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
                    "text": join_units(chunk),
                    # Units let generate.py offer exact sentences with
                    # their section heading, without re-splitting text.
                    "units": [
                        {
                            "text": unit["text"],
                            "section": unit["section"],
                            "evidence": unit["evidence"],
                        }
                        for unit in chunk
                    ],
                }
            )

        oversized = sum(
            len(join_units(chunk).split()) > CHUNK_SIZE
            for chunk in chunks
        )

        print(
            f"{document_id}: "
            f"{len(text.split())} words -> {len(chunks)} chunks"
        )

        if oversized:
            print(
                f"  {oversized} chunk(s) exceed the "
                f"{CHUNK_SIZE}-word target to preserve whole sentences."
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"\nDocuments processed: {len(documents)}")
    print(f"Total chunks: {len(records)}")
    print(
        f"Saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()