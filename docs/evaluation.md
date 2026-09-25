# Evaluation

There are three levels of checks, from fastest to slowest.

| Command | What it checks | Needs models / Ollama |
|---|---|---|
| `pytest` | Chunking, scope filtering, answer validation, API error handling. Retrieval and Ollama are replaced by fakes. | No |
| `python evaluate.py` | 12 hand-written regression cases with required phrases and quality warnings. | Yes |
| `python benchmark.py` | 60 labeled questions: retrieval metrics, abstention, citation validity, latency. | Yes |

## Benchmark dataset

`data/evaluation/benchmark_v1_1.json` (`margin_curated_v1`,
label revision `block_units_v1`):

- 40 answerable questions, each labeled with the chunk(s) that contain
  sufficient evidence.
- 20 unanswerable questions whose answers are not in the collected
  documents (for example, tuition, deadlines, dining hours).
- Each answerable question lists `evidence_phrases`: exact source text
  that is sufficient to answer it. Gold chunks are every in-scope chunk
  containing all of a question's phrases, computed by
  `data/evaluation/relabel.py`.
- Labels are tied to a SHA-256 hash of the chunk corpus. The benchmark
  refuses to run if the corpus has changed since the labels were
  computed. After re-chunking, run `relabel.py`, review the printed
  changes, then run it again with `--write`.
- When the phrases were introduced they reproduced all 40 previously
  reviewed labels exactly on the old chunks. On the new chunks one
  label changed: s34 gained `it_two_step_login_002`, whose overlap now
  contains the same sentence.

## Answer-quality changes (label revision `block_units_v1`)

Problems found in the previous run:

- Sentence splitting ignored line breaks, so headings and list items
  were glued to the next sentence ("eduroam eduroam is ...", a
  57-word run-on of eduroam bullet points, "(2FA)" as a sentence).
- The model chose from up to about 30 sentences and declined only 7 of
  20 unanswerable questions.

Changes:

- `text_units.py`: sentences never cross a line break; headings are
  detected and attached to the sentences below them as context;
  headings, questions, and fragments under 4 words are never offered as
  answers.
- Passage cutoff: `MIN_RERANK_SCORE = -5.0`. On the benchmark, the best
  passage for every answerable question scored at least -3.69, while 5
  unanswerable questions scored below -7. The previous committed value
  (0.0) would have declined 3 answerable questions whose best passage
  scored below 0.
- Only the 8 best sentences (by cross-encoder score with their section
  heading) are offered. On the benchmark the gold sentence ranks at
  worst 5th, so 8 keeps all 40.
- The model writes the fact it needs, chooses one sentence ID (or 0 to
  decline), and copies the answer words. Python declines when the
  words are not in the chosen sentence.

Checks run without the language model (retrieval uses the real
embedding model and reranker):

| Check | Result |
|---|---|
| Retrieval on new chunks (Hit@1 / MRR@7, reranked) | 0.900 / 0.950, unchanged |
| Unanswerable questions declined by the passage cutoff | 5/20 |
| Gold sentence among the 8 offered options | 40/40 |
| Perfect-selector ceiling: correct quotes, false declines | 40/40, 0/40 |
| Reranker's first sentence alone (model says `answers` to all) | 31/40 correct quotes |

### Run 1: model picks one sentence (current design)

Run `20260925T040822998391Z`, `qwen2.5:3b`.

| Metric | Before | Run 1 |
|---|---|---|
| Correct declines on unanswerable questions | 7/20 | **14/20** |
| False declines on answerable questions | 0/40 | 0/40 |
| Answerable questions quoting the right sentence | not measured | 34/40 |
| Median latency after warm-up | 0.82 s | 1.32 s |
| Citation validity | 53/53 | 46/46 |
| Regression checks (`evaluate.py`) | 11/12 | 12/12 |

"Right sentence" was graded by checking each quote against the
reviewed evidence phrases; `benchmark.py` now reports this
automatically as `supported_quotes_gold_evidence`.

Remaining errors: 6 answerable questions quoted a real but wrong
sentence, and 6 unanswerable questions were not declined. Most had one
cause: the model picked sentences that point to information instead
of stating it
("View steps on how to...", "Find out how to add a new device...",
"Note 2: refer to the SGS and Global Policies...", "Instructions for
installing..."). In 5 of the 6 wrong answers the correct sentence was
ranked first by the reranker. The model also copied whole sentences
as its "answer words", so that check rarely rejects anything.

### Run 2: model labels every sentence (reverted)

Run `20260925T042050458857Z`. The model labeled each of the 8 sentences
`answers`, `related`, or `points_elsewhere`, and Python returned the
best-ranked `answers`.

| Metric | Run 1 | Run 2 |
|---|---|---|
| Correct declines on unanswerable questions | 14/20 | 20/20 |
| False declines on answerable questions | 0/40 | 21/40 |
| Answerable questions quoting the right sentence | 34/40 | 16/40 |
| Regression checks (`evaluate.py`) | 12/12 | 7/12 |

Judging each sentence separately made the model label almost nothing
`answers`. It declined everything unanswerable, but also half of the
answerable questions. Run 1's design was restored.

Next ideas to test against Run 1: keep the model's single choice but
prefer the reranker's top sentence when the model's pick is only a
pointer; mark link text as non-answers during collection, where the
HTML still shows what is a link.

## Previous results

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
