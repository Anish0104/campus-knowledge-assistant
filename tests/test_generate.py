"""
Tests for answer_question with retrieval and Ollama replaced by fakes.

No embedding model, reranker, or Ollama server is needed.
"""

import json

import httpx
import pytest

import generate


PASSAGE = {
    "chunk_id": "ms_cs_requirements_006",
    "document_id": "ms_cs_requirements",
    "source_url": "https://example.edu/ms",
    "campus": "New Brunswick",
    "program": "MS Computer Science",
    "text": (
        "The thesis must be approved by the student's thesis committee. "
        "The committee includes the supervisor."
    ),
    "rerank_score": 5.0,
}


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


@pytest.fixture
def fake_retrieval(monkeypatch):
    def fake_rerank_search(question, **kwargs):
        return [dict(PASSAGE)]

    monkeypatch.setattr(generate, "rerank_search", fake_rerank_search)


def fake_ollama(monkeypatch, content, done_reason="stop"):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(content, done_reason)

    monkeypatch.setattr(generate.httpx, "post", fake_post)
    return calls


def test_selected_evidence_is_quoted_verbatim(monkeypatch, fake_retrieval):
    fake_ollama(monkeypatch, json.dumps({"evidence_id": 1}))

    result = generate.answer_question("Who approves my thesis?")

    quote = "The thesis must be approved by the student's thesis committee."
    assert result["claims"] == [
        {"evidence_quote": quote, "source_id": "ms_cs_requirements_006"}
    ]
    assert result["answer"] == f'"{quote}" [ms_cs_requirements_006]'
    assert quote in result["sources"][0]["text"]


def test_model_can_abstain(monkeypatch, fake_retrieval):
    fake_ollama(monkeypatch, json.dumps({"evidence_id": 0}))

    result = generate.answer_question("What time does dining close?")

    assert result["answer"] == generate.FALLBACK_ANSWER
    assert result["claims"] == []
    assert result["sources"] == []


def test_low_scoring_passages_abstain_without_calling_model(monkeypatch):
    low = dict(PASSAGE, rerank_score=generate.MIN_RERANK_SCORE - 1)
    monkeypatch.setattr(generate, "rerank_search", lambda q, **k: [low])
    calls = fake_ollama(monkeypatch, json.dumps({"evidence_id": 1}))

    result = generate.answer_question("Anything?")

    assert result["answer"] == generate.FALLBACK_ANSWER
    assert calls == []


def test_schema_only_allows_offered_ids(monkeypatch, fake_retrieval):
    calls = fake_ollama(monkeypatch, json.dumps({"evidence_id": 0}))

    generate.answer_question("Who approves my thesis?")

    schema = calls[0]["json"]["format"]
    assert schema["properties"]["evidence_id"]["enum"] == [0, 1, 2]


def test_request_uses_configured_model(monkeypatch, fake_retrieval):
    calls = fake_ollama(monkeypatch, json.dumps({"evidence_id": 0}))

    generate.answer_question("Who approves my thesis?")

    assert calls[0]["url"] == f"{generate.OLLAMA_URL}/api/chat"
    assert calls[0]["json"]["model"] == generate.OLLAMA_MODEL


def test_truncated_output_is_rejected(monkeypatch, fake_retrieval):
    fake_ollama(monkeypatch, '{"evidence_id": 1}', done_reason="length")

    with pytest.raises(ValueError):
        generate.answer_question("Who approves my thesis?")


def test_unknown_evidence_id_is_rejected(monkeypatch, fake_retrieval):
    fake_ollama(monkeypatch, json.dumps({"evidence_id": 99}))

    with pytest.raises(ValueError):
        generate.answer_question("Who approves my thesis?")


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        generate.answer_question("   ")


def test_ollama_connection_error_propagates(monkeypatch, fake_retrieval):
    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(generate.httpx, "post", failing_post)

    with pytest.raises(httpx.RequestError):
        generate.answer_question("Who approves my thesis?")
