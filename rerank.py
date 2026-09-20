from functools import lru_cache

from sentence_transformers import CrossEncoder

from search import search


@lru_cache(maxsize=1)
def get_reranker():
    """Load the model once per Python process."""
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
    )


def rerank_search(
    question: str,
    top_k: int = 3,
    candidate_k: int = 7,
    *,
    campus: str | None = None,
    program: str | None = None,
) -> list[dict]:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be blank.")

    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Require candidate_k >= top_k >= 1.")

    candidates = search(
    question,
    top_k=candidate_k,
    campus=campus,
    program=program,
)

    if not candidates:
        return []

    pairs = [
        (question, passage["text"])
        for passage in candidates
    ]

    scores = get_reranker().predict(pairs)

    ranked = [
        {
            **passage,
            "retrieval_rank": rank,
            "rerank_score": float(score),
        }
        for rank, (passage, score) in enumerate(
            zip(candidates, scores),
            start=1,
        )
    ]

    ranked.sort(
        key=lambda passage: passage["rerank_score"],
        reverse=True,
    )

    return ranked[:top_k]


if __name__ == "__main__":
    question = input("Ask a question: ").strip()

    results = rerank_search(question)

    for rank, passage in enumerate(results, start=1):
        print(f"\n--- Reranked result {rank} ---")
        print(f"Chunk: {passage['chunk_id']}")
        print(f"Original rank: {passage['retrieval_rank']}")
        print(f"Rerank score: {passage['rerank_score']:.3f}")
        print(f"Text: {passage['text']}")