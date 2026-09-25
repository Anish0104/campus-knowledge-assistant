import ingest


def test_split_sentences_keeps_abbreviations_together():
    text = "Dr. Smith supervises M.S. students. Theses need approval."

    assert ingest.split_sentences(text) == [
        "Dr. Smith supervises M.S. students.",
        "Theses need approval.",
    ]


def test_split_sentences_normalizes_whitespace():
    assert ingest.split_sentences("  One.\n\n  Two.  ") == ["One.", "Two."]


def test_chunks_contain_only_whole_sentences(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 10)
    monkeypatch.setattr(ingest, "OVERLAP", 4)

    sentences = [f"Sentence number {n} is here." for n in range(6)]
    chunks = ingest.chunk_text(" ".join(sentences))

    assert len(chunks) > 1

    for chunk in chunks:
        for sentence in ingest.split_sentences(chunk):
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
