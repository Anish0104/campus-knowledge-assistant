# Document Sources

All documents are public Rutgers web pages. The catalog with the
metadata used for filtering is `data/documents.json`; the text used for
indexing is in `data/raw/`.

| Document ID | Topic | Scope | Source | Collected | How |
|---|---|---|---|---|---|
| `ms_cs_requirements` | MS CS degree requirements | New Brunswick, MS Computer Science | [cs.rutgers.edu](https://www.cs.rutgers.edu/academics/graduate/m-s-program/degree-requirements) | 2026-09-15 | Main page text copied manually |
| `library_interlibrary_loan` | Interlibrary loan | University-wide | [libraries.rutgers.edu](https://www.libraries.rutgers.edu/find-borrow/interlibrary-loan-borrow-other-libraries) | Not recorded | Copied manually |
| `it_eduroam` | eduroam wireless access | University-wide | [it.rutgers.edu/eduroam](https://it.rutgers.edu/eduroam/) | 2026-09-21 | `collect.py`, reviewed before use |
| `it_two_step_login` | Two-step login with Duo | University-wide | [it.rutgers.edu/two-step-login](https://it.rutgers.edu/two-step-login/) | 2026-09-21 | `collect.py`, reviewed before use |
| `it_microsoft_office` | Microsoft Office access | University-wide | [it.rutgers.edu/microsoft-office](https://it.rutgers.edu/microsoft-office/) | 2026-09-21 | `collect.py`, reviewed before use |

## Notes

- The effective academic year of the MS CS requirements page has not
  been verified. Degree rules can change between catalog years.
- `collect.py` saves each run to `data/staging/` with a manifest and
  content hashes. Staged text is only copied into `data/raw/` after a
  manual review, so a changed page never silently changes the index.
