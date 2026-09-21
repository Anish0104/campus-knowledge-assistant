# Manual evaluation

Corpus: 2 documents, 11 chunks.
Pipeline: semantic retrieval, cross-encoder reranking,
single-passage evidence extraction, exact quote validation.

| Question | Expected source | Observed result |
|---|---|---|
| Who can use Rutgers interlibrary loan? | library_interlibrary_loan_001 | Correct eligibility quote and citation |
| Who approves my thesis? | ms_cs_requirements_006 | Correct thesis committee quote and citation |
| Can I borrow required textbooks through interlibrary loan? | Library textbook restriction | Correct restriction; quote includes a heading fragment |
| What time does Busch dining hall close today? | No supporting source | Correctly declined |
| Who approves an MBA thesis at Rutgers Newark? | No supporting source | FAILED: returned MSCS New Brunswick thesis policy |

Both examples passed. These are development checks, not an
overall accuracy estimate. Campus filtering and questions requiring
multiple passages have not been evaluated.

## Explicit scope filters and relevance cutoff

Using campus/program request fields and a provisional rerank cutoff of 0.0:

- Newark MBA thesis approval: correctly declined.
- New Brunswick MSCS thesis approval: correct quote and citation.
- Library eligibility with Newark/MBA filters: correct quote and citation.

Limitations: scope is not automatically extracted from question text.
The cutoff has only been checked on these development examples.
Quote matching does not guarantee that a quote answers the question.
## Automated development regression checks

Run with: python evaluate.py

Observed result: 6 passed, 0 failed, 0 errors.

Checks cover:
- MSCS thesis approval with supporting evidence.
- Interlibrary loan eligibility.
- Required textbook borrowing restrictions.
- Abstention for unsupported dining hours.
- Abstention for unsupported Newark MBA thesis requirements.
- University-wide library access with Newark MBA filters.

These are known development cases, not a held-out accuracy benchmark.
The checks call Python functions directly, not HTTP or the frontend.
