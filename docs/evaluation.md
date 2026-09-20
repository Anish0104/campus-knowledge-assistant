# Manual evaluation

Corpus: 2 documents, 11 chunks.
Pipeline: semantic retrieval, cross-encoder reranking,
single-passage evidence extraction, exact quote validation.

| Question | Expected source | Observed result |
|---|---|---|
| Who can use Rutgers interlibrary loan? | library_interlibrary_loan_001 | Correct eligibility quote and citation |
| Who approves my thesis? | ms_cs_requirements_006 | Correct thesis committee quote and citation |

Both examples passed. These are development checks, not an
overall accuracy estimate. Campus filtering and questions requiring
multiple passages have not been evaluated.