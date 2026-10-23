import pytest
from pydantic import ValidationError

from generate import (
    EvidenceSelection,
    FALLBACK_ANSWER,
    build_answer,
)


FIRST = "The loan period is twelve weeks."
SECOND = "Articles are delivered electronically."
THIRD = "Books can be delivered to your home."


def candidate(text, chunk_id, document_id="library"):
    passage = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_url": "https://example.edu/library",
        "text": text,
    }
    return {
        "unit": {
            "text": text,
            "section": "Borrowing",
            "evidence": True,
        },
        "passage": passage,
    }


def selection(additional=None, evidence_id=1):
    return EvidenceSelection(
        needed="loan period and delivery",
        evidence_id=evidence_id,
        answer_words=FIRST if evidence_id else "",
        additional_evidence=additional or [],
    )


def test_two_sentences_keep_their_own_citations():
    candidates = {
        1: candidate(FIRST, "library_001"),
        2: candidate(SECOND, "library_002"),
    }
    picked = selection([
        {"evidence_id": 2, "answer_words": SECOND},
    ])

    result = build_answer("Loan period and delivery?", picked, candidates)

    assert len(result["claims"]) == 2
    assert result["claims"][0]["source_id"] == "library_001"
    assert result["claims"][1]["source_id"] == "library_002"
    assert result["answer"] == (
        f'"{FIRST}" [library_001]\n\n'
        f'"{SECOND}" [library_002]'
    )
    assert len(result["sources"]) == 2


def test_three_sentences_are_supported():
    candidates = {
        1: candidate(FIRST, "library_001"),
        2: candidate(SECOND, "library_002"),
        3: candidate(THIRD, "library_003"),
    }
    picked = selection([
        {"evidence_id": 2, "answer_words": SECOND},
        {"evidence_id": 3, "answer_words": THIRD},
    ])

    result = build_answer("Explain borrowing.", picked, candidates)

    assert len(result["claims"]) == 3
    assert len(result["sources"]) == 3


def test_shared_source_is_listed_once():
    first = candidate(FIRST, "library_001")
    second = candidate(SECOND, "library_001")
    source_text = f"{FIRST} {SECOND}"
    first["passage"]["text"] = source_text
    second["passage"]["text"] = source_text

    picked = selection([
        {"evidence_id": 2, "answer_words": SECOND},
    ])

    result = build_answer(
        "Loan period and delivery?",
        picked,
        {1: first, 2: second},
    )

    assert len(result["claims"]) == 2
    assert len(result["sources"]) == 1


def test_invalid_additional_words_decline_the_entire_answer():
    candidates = {
        1: candidate(FIRST, "library_001"),
        2: candidate(SECOND, "library_002"),
    }
    picked = selection([
        {"evidence_id": 2, "answer_words": "Delivery costs ten dollars."},
    ])

    result = build_answer("Loan period and delivery?", picked, candidates)

    assert result["answer"] == FALLBACK_ANSWER
    assert result["claims"] == []
    assert result["sources"] == []


def test_duplicate_evidence_is_rejected():
    picked = selection([
        {"evidence_id": 1, "answer_words": FIRST},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        build_answer(
            "Loan period?",
            picked,
            {1: candidate(FIRST, "library_001")},
        )


def test_unknown_additional_id_is_rejected():
    picked = selection([
        {"evidence_id": 99, "answer_words": SECOND},
    ])

    with pytest.raises(ValueError, match="unknown"):
        build_answer(
            "Loan period and delivery?",
            picked,
            {1: candidate(FIRST, "library_001")},
        )


def test_decline_cannot_contain_additional_evidence():
    picked = selection(
        [{"evidence_id": 1, "answer_words": FIRST}],
        evidence_id=0,
    )

    with pytest.raises(ValueError, match="Decline"):
        build_answer(
            "Unknown question?",
            picked,
            {1: candidate(FIRST, "library_001")},
        )


def test_more_than_three_total_sentences_are_rejected():
    with pytest.raises(ValidationError):
        selection([
            {"evidence_id": 2, "answer_words": SECOND},
            {"evidence_id": 3, "answer_words": THIRD},
            {"evidence_id": 4, "answer_words": "Another sentence."},
        ])


def test_additional_quote_must_exist_in_its_source():
    invalid = candidate(SECOND, "library_002")
    invalid["passage"]["text"] = "This source contains something else."
    picked = selection([
        {"evidence_id": 2, "answer_words": SECOND},
    ])

    with pytest.raises(ValueError, match="mapping"):
        build_answer(
            "Loan period and delivery?",
            picked,
            {
                1: candidate(FIRST, "library_001"),
                2: invalid,
            },
        )

