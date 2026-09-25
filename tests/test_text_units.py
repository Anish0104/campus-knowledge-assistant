from text_units import classify_headings, join_units, split_sentences, text_units


def test_split_sentences_keeps_abbreviations_together():
    text = "Dr. Smith supervises M.S. students. Theses need approval."

    assert split_sentences(text) == [
        "Dr. Smith supervises M.S. students.",
        "Theses need approval.",
    ]


def test_split_sentences_normalizes_whitespace():
    assert split_sentences("  One.\n\n  Two.  ") == ["One.", "Two."]


def test_sentences_never_cross_line_breaks():
    text = (
        "Why use it?\n\n"
        "Connect to free WiFi at member universities\n\n"
        "Install on your device and log in with your NetID\n\n"
        "Visitors can connect here."
    )

    texts = [unit["text"] for unit in text_units(text)]

    assert texts == [
        "Why use it?",
        "Connect to free WiFi at member universities",
        "Install on your device and log in with your NetID",
        "Visitors can connect here.",
    ]


def test_title_is_not_glued_to_first_sentence():
    units = text_units("eduroam\n\neduroam is a free and secure network.")

    assert units[0]["text"] == "eduroam"
    assert units[0]["heading"] is True
    assert units[1]["text"] == "eduroam is a free and secure network."
    assert units[1]["section"] == "eduroam"


def test_heading_kinds():
    lines = [
        "Page title",
        "A first sentence about the page.",
        "Essay Option:",
        "The essay must be approved.",
        "- Two from Category A",
        "Why use it?",
        "Connect to free WiFi at member universities",
        "Security exceeds typical commercial hotspots",
        "Visitors from member institutions can connect here",
    ]

    assert classify_headings(lines) == [
        "label",   # opens the document
        None,
        "colon",   # short line ending in a colon
        None,
        None,      # list items are never headings
        "label",   # short question
        None,      # unpunctuated list-like lines stay content
        None,
        None,
    ]


def test_label_heading_only_applies_to_next_line():
    text = (
        "Office video tutorials\n"
        "Sign in to LinkedIn Learning for video tutorials.\n"
        "Licenses stay active while students are enrolled."
    )

    units = text_units(text)

    assert units[1]["section"] == "Office video tutorials"
    assert units[2]["section"] is None


def test_colon_heading_applies_until_next_colon_heading():
    text = (
        "Thesis Option:\n"
        "The student must write a thesis.\n"
        "The thesis must be approved by the committee.\n"
        "Degree Completion Processing:\n"
        "Forms are due on time."
    )

    sections = [unit["section"] for unit in text_units(text)]

    assert sections == [
        "Thesis Option:",
        "Thesis Option:",
        "Thesis Option:",
        "Degree Completion Processing:",
        "Degree Completion Processing:",
    ]


def test_headings_questions_and_fragments_are_not_evidence():
    text = (
        "Getting help with Duo\n"
        "Wondering about international travel? Get answers and assistance.\n"
        "What is two-factor authentication? (2FA)"
    )

    evidence = {
        unit["text"]: unit["evidence"]
        for unit in text_units(text)
    }

    assert evidence == {
        "Getting help with Duo": False,           # heading
        "Wondering about international travel?": False,  # question
        "Get answers and assistance.": True,
        "What is two-factor authentication?": False,
        "(2FA)": False,                           # fragment
    }


def test_join_units_restores_line_breaks():
    text = "Title\nFirst sentence. Second sentence.\nThird line."
    units = text_units(text)

    assert join_units(units) == text


def test_short_list_before_a_heading_stays_content():
    lines = [
        "Free WiFi at members",
        "Uses your NetID",
        "Faster than hotspots",
        "Works on phones",
        "Get help",
        "Contact the Help Desk for problems.",
    ]

    kinds = classify_headings(lines)

    assert kinds[4] == "label"
    assert kinds[3] == "label" and kinds[2] == "label"
    assert kinds[1] is None
