import ingest
from text_units import split_sentences, text_units


def test_chunks_contain_only_whole_sentences(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 10)
    monkeypatch.setattr(ingest, "OVERLAP", 4)

    sentences = [f"Sentence number {n} is here." for n in range(6)]
    chunks = ingest.chunk_text(" ".join(sentences))

    assert len(chunks) > 1

    for chunk in chunks:
        for sentence in split_sentences(chunk):
            assert sentence in sentences


def test_chunks_overlap_by_trailing_sentences(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 10)
    monkeypatch.setattr(ingest, "OVERLAP", 5)

    text = "A b c d e. F g h i j. K l m n o."
    chunks = ingest.chunk_text(text)

    assert chunks == [
        "A b c d e. F g h i j.",
        "F g h i j. K l m n o.",
    ]


def test_long_sentence_is_kept_intact(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 5)
    monkeypatch.setattr(ingest, "OVERLAP", 0)

    long_sentence = "This sentence has more than five words in it."
    chunks = ingest.chunk_text(long_sentence)

    assert chunks == [long_sentence]


def test_chunk_keeps_line_breaks(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 100)

    text = "Title\nFirst sentence here. Second one.\nThird line."

    assert ingest.chunk_text(text) == [text]


def test_chunk_never_ends_with_a_heading(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 12)
    monkeypatch.setattr(ingest, "OVERLAP", 0)

    text = (
        "One two three four five six seven eight.\n"
        "Next Topic Title\n"
        "Nine ten eleven twelve thirteen fourteen fifteen."
    )

    chunks = ingest.chunk_units(text_units(text))
    first, second = chunks

    assert first[-1]["text"] == "One two three four five six seven eight."
    assert second[0]["text"] == "Next Topic Title"
    assert second[0]["heading"] is True
