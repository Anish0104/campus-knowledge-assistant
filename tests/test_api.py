import httpx
import pytest
from fastapi.testclient import TestClient

import api
import search


@pytest.fixture
def client():
    return TestClient(api.app)


def test_homepage_serves_the_interface(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Margin" in response.text


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer(client, monkeypatch):
    def fake_answer(question, campus=None, program=None):
        return {
            "question": question,
            "answer": "ok",
            "claims": [],
            "sources": [],
            "scope": [campus, program],
        }

    monkeypatch.setattr(api, "answer_question", fake_answer)

    response = client.post(
        "/ask",
        json={
            "question": "  Who approves my thesis?  ",
            "campus": " New   Brunswick ",
            "program": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Who approves my thesis?"
    assert body["scope"] == ["New Brunswick", None]


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_blank_questions(client, question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (httpx.ReadTimeout("slow"), 504),
        (httpx.ConnectError("refused"), 503),
        (ValueError("bad output"), 502),
    ],
)
def test_ask_maps_errors_to_status_codes(client, monkeypatch, error, status):
    def failing_answer(*args, **kwargs):
        raise error

    monkeypatch.setattr(api, "answer_question", failing_answer)

    response = client.post("/ask", json={"question": "Anything?"})

    assert response.status_code == status


def test_scope_filter_keeps_university_wide_documents():
    library = {"campus": "University-wide", "program": "Not program-specific"}
    mscs = {"campus": "New Brunswick", "program": "MS Computer Science"}

    assert search.in_scope(library, "Newark", "MBA")
    assert not search.in_scope(mscs, "Newark", "MBA")
    assert search.in_scope(mscs, "new brunswick", "ms computer science")
    assert search.in_scope(mscs, None, None)
