"""
Split document text into units: sentences that never cross a line break.

Web pages flatten into many short lines (headings, list items, link
labels) that have no sentence-ending punctuation. Joining them with
the next sentence produces run-on "sentences" such as a heading glued
to a paragraph. Here every line is its own block, and sentences are
only detected inside a line.

Each unit records:
- text: the exact unit text (whitespace-normalized).
- line: index of the source line, so chunks can rebuild line breaks.
- heading: whether the line looks like a heading.
- section: the heading that applies to the unit, if any, used as
  context when the unit is offered to the model.
- evidence: whether the unit may be offered as an answer. Headings,
  questions, and fragments shorter than MIN_EVIDENCE_WORDS are not.
"""

import re


# A unit shorter than this cannot be offered as an answer.
# Fragments such as "(2FA)" or "Get help" carry no answerable fact.
MIN_EVIDENCE_WORDS = 4

# Short unpunctuated lines at most this long may be headings.
MAX_HEADING_WORDS = 6

# At most this many short lines stacked directly above a heading are
# treated as headings too ("Help & Support" above "Get help").
MAX_HEADING_CHAIN = 2

# Lines ending with a colon at most this long are headings.
MAX_COLON_HEADING_WORDS = 10

_BOUNDARY = re.compile(r"""[.!?]+["')\]]*(?=\s|$)""")
_ABBREVIATION = re.compile(
    r"(?:\b(?:M\.S|B\.S|Ph\.D|Dr|Mr|Mrs|Ms|Prof|e\.g|i\.e)|\b[A-Z])\.$"
)
_ENDS_SENTENCE = re.compile(r"""[.!?]["')\]]*$""")
_ENDS_STATEMENT = re.compile(r"""[.!]["')\]]*$""")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """
    Detect sentence boundaries without rewriting source wording.

    Whitespace is normalized. Common abbreviations and initials are
    protected. This is a heuristic, applied to one line at a time.
    """
    text = normalize_space(text)

    if not text:
        return []

    sentences = []
    start = 0

    for match in _BOUNDARY.finditer(text):
        end = match.end()

        # Avoid splitting after common abbreviations or initials.
        if _ABBREVIATION.search(text[:end]):
            continue

        sentence = text[start:end].strip()

        if sentence:
            sentences.append(sentence)

        start = end

    remainder = text[start:].strip()

    if remainder:
        sentences.append(remainder)

    return sentences


def source_lines(text: str) -> list[str]:
    """Non-empty lines with normalized whitespace."""
    return [
        normalize_space(line)
        for line in text.splitlines()
        if line.strip()
    ]


def _short_label(line: str) -> bool:
    """A short line that is not a sentence-ending statement or list item."""
    return (
        not line.startswith("- ")
        and len(line.split()) <= MAX_HEADING_WORDS
        and not _ENDS_STATEMENT.search(line)
        and len(split_sentences(line)) == 1
    )


def classify_headings(lines: list[str]) -> list[str | None]:
    """
    Label each line "colon", "label", or None (content).

    - "colon": a short line ending in a colon ("Essay Option:"). Real
      document structure; it applies to everything until the next
      colon heading.
    - "label": a short line without a final period that opens the
      document, is a short question ("Why use eduroam?"), is followed
      by ordinary sentences ("Enrollment guide" -> "Signing up ..."),
      or sits directly above another heading (at most two stacked
      lines). These are web page card titles; they only describe the
      line right after them.
    - List items ("- ...") are never headings.
    """
    kinds: list[str | None] = [None] * len(lines)

    # How many "above another heading" steps made a line a heading.
    # Capped so a run of short list items before a heading does not
    # turn into headings.
    chain = [0] * len(lines)

    # Walk backwards so each line knows whether the next is a heading.
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        words = line.split()
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        next_kind = kinds[index + 1] if index + 1 < len(lines) else None

        if line.startswith("- "):
            continue

        if line.endswith(":") and len(words) <= MAX_COLON_HEADING_WORDS:
            kinds[index] = "colon"
            continue

        if not _short_label(line):
            continue

        if (
            index == 0
            or line.endswith("?")
            or (next_line and _ENDS_SENTENCE.search(next_line))
        ):
            kinds[index] = "label"
        elif next_kind is not None and chain[index + 1] < MAX_HEADING_CHAIN:
            kinds[index] = "label"
            chain[index] = chain[index + 1] + 1

    return kinds


def text_units(text: str) -> list[dict]:
    lines = source_lines(text)
    kinds = classify_headings(lines)
    units = []

    colon_section = None
    label = None
    label_line = None

    for index, line in enumerate(lines):
        kind = kinds[index]

        if kind == "colon":
            colon_section = line
            label = None
        elif kind == "label":
            label = line
            label_line = index

        if kind:
            section = line
        elif label is not None and label_line == index - 1:
            section = label
        else:
            section = colon_section

        for sentence in split_sentences(line):
            units.append({
                "text": sentence,
                "line": index,
                "heading": kind is not None,
                "section": section,
                "evidence": (
                    kind is None
                    and not sentence.endswith("?")
                    and len(sentence.split()) >= MIN_EVIDENCE_WORDS
                ),
            })

    return units


def join_units(units: list[dict]) -> str:
    """Rebuild text from units, keeping the original line breaks."""
    parts = []
    previous_line = None

    for unit in units:
        if previous_line is not None:
            parts.append("\n" if unit["line"] != previous_line else " ")

        parts.append(unit["text"])
        previous_line = unit["line"]

    return "".join(parts)
