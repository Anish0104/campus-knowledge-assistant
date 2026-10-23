import json
import httpx
import pytest
from fastapi.testclient import TestClient

import api
import generate
from providers import GroqProvider, OllamaProvider, get_provider


APPROVAL = "The thesis must be approved by the student's thesis committee."

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
        {"text": "Thesis Option:", "section": "Thesis Option:", "evidence": False},
        {
            "text": APPROVAL,
            "section": "Thesis Option:",
            "evidence": True,
        },
        {
            "text": "The committee includes the supervisor.",
            "section": "Thesis Option:",
            "evidence": True,
        },
    ],
    "rerank_score": 5.0,
}


class FakeResponse:
    def __init__(
        self,
        payload: dict,
        status_code: int = 200,
        headers: dict | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, headers=self.headers),
            )

    def json(self):
        return self._payload


class FakeReranker:
    def predict(self, pairs):
        return [5.0 for _ in pairs]


@pytest.fixture(autouse=True)
def fake_retrieval(monkeypatch):
    monkeypatch.setattr(generate, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(generate, "rerank_search", lambda q, **k: [dict(THESIS)])


def make_groq_payload(content: str, finish_reason: str = "stop", refusal: str | None = None) -> dict:
    message = {"content": content}
    if refusal:
        message["refusal"] = refusal
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": message,
            }
        ]
    }


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"
    assert provider.model_name == generate.OLLAMA_MODEL


def test_unknown_provider_raises_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_provider()


def test_groq_request_construction_and_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    calls = []
    expected_content = json.dumps({
        "needed": "thesis committee approval",
        "evidence_id": 1,
        "answer_words": "thesis committee",
    })

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(make_groq_payload(expected_content))

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = get_provider()
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"
    assert provider.model_name == "llama-3.3-70b-versatile"

    content = provider.generate_evidence_selection(
        system_prompt="Test System Prompt",
        user_content='{"question": "test"}',
        schema={"type": "object", "properties": {"evidence_id": {"type": "integer"}}},
        multi=False,
    )

    assert content == expected_content
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer gsk_test_mock_key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["json"]["model"] == "llama-3.3-70b-versatile"
    assert call["json"]["temperature"] == 0
    assert call["json"]["max_completion_tokens"] == 2048
    assert call["json"]["response_format"]["type"] == "json_schema"
    assert call["json"]["response_format"]["json_schema"]["strict"] is True


def test_groq_valid_selection_reaches_quote_validation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    groq_content = json.dumps({
        "needed": "who approves the thesis",
        "evidence_id": 1,
        "answer_words": "thesis committee",
    })

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(make_groq_payload(groq_content)),
    )

    result = generate.answer_question("Who approves my thesis?")
    assert result["answer"] == f'"{APPROVAL}" [ms_cs_requirements_006]'
    assert result["claims"][0]["evidence_quote"] == APPROVAL
    assert result["claims"][0]["source_id"] == "ms_cs_requirements_006"
    assert result["claims"][0]["answer_words"] == "thesis committee"


def test_groq_unknown_evidence_id_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    groq_content = json.dumps({
        "needed": "who approves the thesis",
        "evidence_id": 99,
        "answer_words": "thesis committee",
    })

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(make_groq_payload(groq_content)),
    )

    with pytest.raises(ValueError, match="unknown evidence ID"):
        generate.answer_question("Who approves my thesis?")


def test_groq_invalid_answer_words_declined(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    groq_content = json.dumps({
        "needed": "who approves the thesis",
        "evidence_id": 1,
        "answer_words": "not present in text at all",
    })

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(make_groq_payload(groq_content)),
    )

    result = generate.answer_question("Who approves my thesis?")
    assert result["answer"] == generate.FALLBACK_ANSWER
    assert result["claims"] == []
    assert result["sources"] == []
    assert "declined_reason" in result["selection"]


def test_groq_truncation_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(make_groq_payload("{}", finish_reason="length")),
    )

    with pytest.raises(ValueError, match="did not complete normally"):
        generate.answer_question("Who approves my thesis?")


def test_groq_refusal_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(make_groq_payload("", refusal="Safety violation")),
    )

    with pytest.raises(ValueError, match="Groq refused request"):
        generate.answer_question("Who approves my thesis?")


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"finish_reason": "stop", "message": {}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]},
    ],
)
def test_groq_malformed_and_empty_responses(monkeypatch, payload):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(ValueError):
        generate.answer_question("Who approves my thesis?")


def test_groq_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")

    def unexpected_post(*args, **kwargs):
        raise AssertionError("Network call was made when API key was missing.")

    monkeypatch.setattr(httpx, "post", unexpected_post)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        generate.answer_question("Who approves my thesis?")


def test_groq_blank_model_raises_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
    monkeypatch.setenv("GROQ_MODEL", "   ")

    with pytest.raises(ValueError, match="GROQ_MODEL cannot be blank"):
        generate.answer_question("Who approves my thesis?")


def test_groq_authentication_failure_not_retried(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "invalid_key")

    call_count = 0

    def failing_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse({}, status_code=401)

    monkeypatch.setattr(httpx, "post", failing_post)

    with pytest.raises(httpx.HTTPStatusError):
        generate.answer_question("Who approves my thesis?")

    # Verify authentication errors are NOT retried
    assert call_count == 1


def test_groq_timeout_propagates(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    def timeout_post(*args, **kwargs):
        raise httpx.ReadTimeout("Request timed out", request=httpx.Request("POST", "https://api.groq.com"))

    monkeypatch.setattr(httpx, "post", timeout_post)

    with pytest.raises(httpx.TimeoutException):
        generate.answer_question("Who approves my thesis?")


def test_groq_rate_limiting_retries_and_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    valid_content = json.dumps({
        "needed": "who approves the thesis",
        "evidence_id": 1,
        "answer_words": "thesis committee",
    })

    call_count = 0

    def rate_limited_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return FakeResponse(
                {"error": {"message": "Rate limit reached"}},
                status_code=429,
                headers={"Retry-After": "0.01"},
            )
        return FakeResponse(make_groq_payload(valid_content))

    monkeypatch.setattr(httpx, "post", rate_limited_post)

    result = generate.answer_question("Who approves my thesis?")
    assert result["answer"] == f'"{APPROVAL}" [ms_cs_requirements_006]'
    assert call_count == 2


def test_groq_rate_limiting_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")

    call_count = 0

    def always_429(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse(
            {"error": {"message": "Rate limit reached"}},
            status_code=429,
            headers={"Retry-After": "0.01"},
        )

    monkeypatch.setattr(httpx, "post", always_429)

    with pytest.raises(httpx.HTTPStatusError):
        generate.answer_question("Who approves my thesis?")

    # 1 initial call + 3 retries = 4 total calls
    assert call_count == 4


def test_groq_rate_limit_exceeds_wait_budget_stops_immediately(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
    monkeypatch.setenv("GROQ_WAIT_BUDGET_SECONDS", "10.0")

    call_count = 0

    def excessive_delay_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse(
            {"error": {"message": "Rate limit reached"}},
            status_code=429,
            headers={"Retry-After": "120.0"},
        )

    monkeypatch.setattr(httpx, "post", excessive_delay_post)

    with pytest.raises(ValueError, match="exceeds remaining wait budget"):
        generate.answer_question("Who approves my thesis?")

    # Verify we did NOT retry or shorten delay
    assert call_count == 1


def test_provider_metadata_accurate(monkeypatch):
    client = TestClient(api.app)

    # Test Ollama metadata
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    resp_ollama = client.get("/health")
    assert resp_ollama.status_code == 200
    data_ollama = resp_ollama.json()
    assert data_ollama["status"] == "ok"
    assert data_ollama["provider"] == "ollama"
    assert data_ollama["model"] == generate.OLLAMA_MODEL

    # Test Groq metadata
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    resp_groq = client.get("/health")
    assert resp_groq.status_code == 200
    data_groq = resp_groq.json()
    assert data_groq["status"] == "ok"
    assert data_groq["provider"] == "groq"
    assert data_groq["model"] == "openai/gpt-oss-20b"

