# Evaluation

There are three levels of checks, from fastest to slowest.

| Command | What it checks | Needs models / Ollama |
|---|---|---|
| `pytest` | Chunking, scope filtering, answer validation, API error handling. Retrieval and Ollama are replaced by fakes. | No |
| `python evaluate.py` | 12 hand-written regression cases with required phrases and quality warnings. | Yes |
| `python benchmark.py` | 60 labeled questions: retrieval metrics, abstention, citation validity, latency. | Yes |

## Benchmark dataset

`data/evaluation/benchmark_v1_1.json` (`margin_curated_v1`,
label revision `sentence_chunks_v1`):

- 40 answerable questions, each labeled with the chunk(s) that contain
  sufficient evidence.
- 20 unanswerable questions whose answers are not in the collected
  documents (for example, tuition, deadlines, dining hours).
- Labels are tied to a SHA-256 hash of the chunk corpus. The benchmark
  refuses to run if the corpus has changed since the labels were
  reviewed.

## Latest recorded results

Run `20260925T005350447655Z` (2026-09-25, macOS arm64, Python 3.14).

This run used a working copy between commits `c41a918` and `8434484`
with `min_rerank_score = -4.0`. The committed `generate.py` uses `0.0`,
so these numbers need to be regenerated for the current code.

**Retrieval (40 answerable questions, 7 candidates):**

| Ranking | Hit@1 | Hit@5 | Recall@5 | MRR@7 |
|---|---|---|---|---|
| Embedding only | 0.775 | 1.00 | 1.00 | 0.877 |
| + Cross-encoder rerank | **0.900** | 1.00 | 1.00 | **0.950** |

**Answers (60 questions):**

| Metric | Result |
|---|---|
| Verbatim citation validity | 53/53 (100%) |
| False abstention on answerable questions | 0/40 (0%) |
| Correct abstention on unanswerable questions | **7/20 (35%)** |
| Answer errors | 0/60 |
| Median / p95 latency (after warm-up) | 0.82 s / 1.23 s |

**Main weakness:** 13 of the 20 unanswerable questions returned a quote
instead of declining. Every quote was real, but a real quote on the
right topic is not an answer. For example, "What is the tuition per
credit for MS Computer Science?" returned the sentence requiring thesis
students to register for six credits, and "What is the overdue fine per
day for an EZBorrow book?" returned the EZBorrow loan period. Improving
abstention is the top priority for the answer stage.

## Limitations

- Curated questions written from the same five documents. This is not
  an independent measure of general accuracy.
- Supported-answer correctness still requires manual review of
  `review.csv` in each run folder.
- Latency is sequential Python calls after warm-up, not HTTP latency or
  a load test.
- Citation validity shows a quote exists in the source; it does not
  show that the quote answers the question.

## History

Early manual checks (2 documents, 11 chunks) found that the Newark MBA
thesis question returned the New Brunswick MSCS thesis policy. Adding
explicit campus/program filters and a rerank score cutoff fixed that
case. These checks became the first cases in `evaluate.py`.
