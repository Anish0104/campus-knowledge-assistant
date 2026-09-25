# Initial Retrieval Checks

> Historical notes from the first retrieval experiment. The current
> setup (sentence-aware chunking, cross-encoder reranking, five
> documents) is described in the README and `docs/evaluation.md`.

Model: sentence-transformers/all-MiniLM-L6-v2
Corpus: One Rutgers MS CS requirements document, seven chunks
Chunking: 150 words, 30-word overlap
Search: Cosine similarity using normalized embeddings

| Question | Answer-containing chunk | Rank |
|---|---|---|
| How many credits are required for the MS CS degree? | 001 | 1 |
| Who needs to approve my thesis? | 006 | 2 |

## Observation

The highest similarity score does not guarantee that a passage
answers the question. Retrieving three passages included the
answer for both examples.

These are two manual checks, not a comprehensive evaluation.
The current corpus covers only MS CS requirements.

## Possible improvement

Experiment with sentence-aware chunking or a reranker, then
compare results on a larger set of labeled questions.
