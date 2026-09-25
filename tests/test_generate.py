"""
Tests for answer_question with retrieval, reranking, and Ollama
replaced by fakes. No models or Ollama server are needed.
"""

import json

import httpx
import pytest

import generate


def unit(text, section=None, evidence=True):
    return {"text": text, "section": section, "evidence": evidence}


THESIS = {
    "chunk_id": "ms_cs_requirements_006",
    "document_id": "ms_cs_requirements",
    "source_url": "https://example.edu/ms",
    "campus": "New Brunswick",
    "program": "MS Computer Science",
    "topic": "Degree requirements",
    "text": (
        "Thesis Option:\n"
        "The thesis must be approved by the student's thesis committee. "
        "The committee includes the supervisor."
    ),
    "units": [
        unit("Thesis Option:", "Thesis Option:", evidence=False),
        unit(
            "The thesis must be approved by the student's thesis committee.",
            "Thesis Option:",
        ),
        unit("The committee includes the supervisor.", "Thesis Option:"),
    ],
    "rerank_score": 5.0,
}

APPROVAL = "The thesis must be approved by the student's thesis committee."


class FakeResponse:
    def __init__(self, content: str, done_reason: str = "stop"):
        self._payload = {
            "message": {"content": content},
            "done_reason": done_reason,
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeReranker:
    """Scores a sentence by how many question words it contains."""

    def predict(self, pairs):
        return [
            float(sum(word in text.lower() for word in question.lower().split()))
            for question, text in pairs
        ]


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(generate, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(
        generate, "rerank_search", lambda question, **kwargs: [dict(THESIS)]
    )


def fake_ollama(monkeypatch, evidence_id, answer_words="", done_reason="stop"):
    calls = []
    content = json.dumps({
        "needed": "who approves the thesis",
        "evidence_id": evidence_id,
        "answer_words": answer_words,
    })

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(content, done_reason)

    monkeypatch.setattr(generate.httpx, "post", fake_post)
    return calls


def offered(calls):
    body = json.loads(calls[0]["json"]["messages"][1]["content"])
    return body["evidence_options"]


def approval_id(calls):
    return next(
        option["evidence_id"]
        for option in offered(calls)
        if option["text"] == APPROVAL
    )


def test_selected_evidence_is_quoted_verbatim(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)
    generate.answer_question("Who approves my thesis?")
    chosen = approval_id(calls)

    fake_ollama(monkeypatch, chosen, "thesis committee")
    result = generate.answer_question("Who approves my thesis?")

    assert result["answer"] == f'"{APPROVAL}" [ms_cs_requirements_006]'
    assert result["claims"][0]["evidence_quote"] == APPROVAL
    assert result["claims"][0]["section"] == "Thesis Option:"
    assert result["claims"][0]["answer_words"] == "thesis committee"
    assert APPROVAL in result["sources"][0]["text"]
    assert "units" not in result["sources"][0]


def test_answer_words_are_matched_loosely(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)
    generate.answer_question("Who approves my thesis?")
    chosen = approval_id(calls)

    fake_ollama(monkeypatch, chosen, "  The Student’s Thesis   Committee. ")
    result = generate.answer_question("Who approves my thesis?")

    assert result["claims"]


def test_answer_words_missing_from_evidence_declines(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)
    generate.answer_question("Who approves my thesis?")
    chosen = approval_id(calls)

    fake_ollama(monkeypatch, chosen, "the graduate director")
    result = generate.answer_question("Who approves my thesis?")

    assert result["answer"] == generate.FALLBACK_ANSWER
    assert result["claims"] == []
    assert result["sources"] == []
    assert "declined_reason" in result["selection"]


@pytest.mark.parametrize("answer_words", ["", "   ", "."])
def test_empty_answer_words_decline(monkeypatch, answer_words):
    fake_ollama(monkeypatch, 1, answer_words)

    result = generate.answer_question("Who approves my thesis?")

    assert result["answer"] == generate.FALLBACK_ANSWER


def test_model_can_abstain(monkeypatch):
    fake_ollama(monkeypatch, 0)

    result = generate.answer_question("What time does dining close?")

    assert result["answer"] == generate.FALLBACK_ANSWER
    assert result["claims"] == []
    assert result["sources"] == []
    assert result["selection"]["evidence_id"] == 0


def test_low_scoring_passages_abstain_without_calling_model(monkeypatch):
    low = dict(THESIS, rerank_score=generate.MIN_RERANK_SCORE - 1)
    monkeypatch.setattr(generate, "rerank_search", lambda q, **k: [low])
    calls = fake_ollama(monkeypatch, 1, "thesis committee")

    result = generate.answer_question("Anything?")

    assert result["answer"] == generate.FALLBACK_ANSWER
    assert calls == []


def test_headings_are_never_offered(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)

    generate.answer_question("Who approves my thesis?")

    texts = [option["text"] for option in offered(calls)]
    assert "Thesis Option:" not in texts
    assert all(option["section"] == "Thesis Option:" for option in offered(calls))


def test_overlapping_sentences_are_offered_once(monkeypatch):
    overlap = dict(THESIS, chunk_id="ms_cs_requirements_005", rerank_score=4.0)
    monkeypatch.setattr(
        generate, "rerank_search", lambda q, **k: [dict(THESIS), overlap]
    )
    calls = fake_ollama(monkeypatch, 0)

    generate.answer_question("Who approves my thesis?")

    texts = [option["text"] for option in offered(calls)]
    assert len(texts) == len(set(texts)) == 2


def test_options_are_limited_and_ranked(monkeypatch):
    many = dict(THESIS)
    many["units"] = [
        unit(f"Filler sentence number {n} here.") for n in range(12)
    ] + [unit(APPROVAL)]
    many["text"] = " ".join(u["text"] for u in many["units"])
    monkeypatch.setattr(generate, "rerank_search", lambda q, **k: [many])
    calls = fake_ollama(monkeypatch, 0)

    generate.answer_question("Who approves the thesis committee?")

    options = offered(calls)
    assert len(options) == generate.MAX_EVIDENCE_OPTIONS
    assert options[0]["text"] == APPROVAL


def test_schema_only_allows_offered_ids_and_asks_for_needed_first(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)

    generate.answer_question("Who approves my thesis?")

    schema = calls[0]["json"]["format"]
    assert schema["properties"]["evidence_id"]["enum"] == [0, 1, 2]
    assert list(schema["properties"])[:3] == [
        "needed",
        "evidence_id",
        "answer_words",
    ]


def test_request_uses_configured_model(monkeypatch):
    calls = fake_ollama(monkeypatch, 0)

    generate.answer_question("Who approves my thesis?")

    assert calls[0]["url"] == f"{generate.OLLAMA_URL}/api/chat"
    assert calls[0]["json"]["model"] == generate.OLLAMA_MODEL
    assert calls[0]["json"]["options"]["temperature"] == 0


def test_truncated_output_is_rejected(monkeypatch):
    fake_ollama(monkeypatch, 1, "x", done_reason="length")

    with pytest.raises(ValueError):
        generate.answer_question("Who approves my thesis?")


def test_unknown_evidence_id_is_rejected(monkeypatch):
    fake_ollama(monkeypatch, 99, "thesis committee")

    with pytest.raises(ValueError):
        generate.answer_question("Who approves my thesis?")


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        generate.answer_question("   ")


def test_ollama_connection_error_propagates(monkeypatch):
    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(generate.httpx, "post", failing_post)

    with pytest.raises(httpx.RequestError):
        generate.answer_question("Who approves my thesis?")
