import numpy as np
import pytest

import search


RECORDS = [
    {"chunk_id": "a", "campus": "New Brunswick", "program": "MS Computer Science", "text": "A"},
    {"chunk_id": "b", "campus": "University-wide", "program": "Not program-specific", "text": "B"},
    {"chunk_id": "c", "campus": "Newark", "program": "MBA", "text": "C"},
]

EMBEDDINGS = np.array([
    [1.0, 0.0],
    [0.6, 0.8],
    [0.0, 1.0],
])


class FakeEmbedder:
    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0])


@pytest.fixture(autouse=True)
def fake_index(monkeypatch):
    monkeypatch.setattr(search, "load_index", lambda: (RECORDS, EMBEDDINGS))
    monkeypatch.setattr(search, "get_embedder", lambda: FakeEmbedder())


def test_results_are_ranked_by_similarity():
    results = search.search("question", top_k=3)

    assert [r["chunk_id"] for r in results] == ["a", "b", "c"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_scope_filter_runs_before_ranking():
    results = search.search("question", top_k=3, campus="Newark", program="MBA")

    assert [r["chunk_id"] for r in results] == ["b", "c"]


def test_top_k_limits_results():
    assert len(search.search("question", top_k=1)) == 1


@pytest.mark.parametrize(("question", "top_k"), [("  ", 3), ("q", 0)])
def test_invalid_input_is_rejected(question, top_k):
    with pytest.raises(ValueError):
        search.search(question, top_k=top_k)
