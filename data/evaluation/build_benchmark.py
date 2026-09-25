import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "benchmark.json"

# Each listed chunk contains sufficient evidence for the question.
# Labels should be reviewed before running the benchmark.
SUPPORTED = [
    ("ms_cs_requirements", [1],
     "How many credits are required for the MS Computer Science degree?",
     "36 credits, equivalent to 12 courses.", True),
    ("ms_cs_requirements", [1],
     "How many Category A courses are explicitly required?",
     "Two Category A courses.", False),
    ("ms_cs_requirements", [1],
     "How many Category B courses are explicitly required?",
     "Two Category B courses.", False),
    ("ms_cs_requirements", [1],
     "What is the maximum number of independent study courses among the four elective courses?",
     "At most two of those four courses.", False),
    ("ms_cs_requirements", [2],
     "What minimum GPA does the MSCS document require?",
     "A minimum GPA of 3.", False),
    ("ms_cs_requirements", [2],
     "How many grades of C or lower does the MSCS document permit?",
     "No more than two.", False),
    ("ms_cs_requirements", [3],
     "What language must the essay or thesis use?",
     "English, except computer-language portions with English documentation.", False),
    ("ms_cs_requirements", [3],
     "Which sections must the MSCS essay contain?",
     "Title, authors, abstract, and references.", False),
    ("ms_cs_requirements", [3],
     "What grade must a previously submitted course paper have received to be used as the essay?",
     "B or higher.", False),
    ("ms_cs_requirements", [4],
     "What must I provide if my essay was a group assignment?",
     "Cover pages detailing my contributions, along with the project report.", False),
    ("ms_cs_requirements", [4],
     "Are presentation slides required with an essay submission?",
     "Yes, corresponding presentation slides are required.", False),
    ("ms_cs_requirements", [4],
     "Do students receive extra credit for preparing the essay?",
     "No.", False),
    ("ms_cs_requirements", [4],
     "Who must approve the MSCS essay?",
     "A Computer Science Graduate Faculty member, followed by the MS Director.", True),
    ("ms_cs_requirements", [4],
     "How many credits of 198:704-706 must a thesis student register for?",
     "Exactly six credits.", False),
    ("ms_cs_requirements", [6],
     "Who makes up the thesis committee?",
     "The thesis supervisor and two other faculty members selected in consultation with the supervisor.", True),
    ("ms_cs_requirements", [6],
     "How should a student choose a thesis topic?",
     "By mutual agreement with the faculty supervisor.", False),

    ("library_interlibrary_loan", [1],
     "How will I know when an interlibrary loan item is ready?",
     "An email will notify me when it is ready for pickup or download.", False),
    ("library_interlibrary_loan", [1, 2],
     "How are requested articles and book chapters delivered?",
     "Electronically.", False),
    ("library_interlibrary_loan", [1, 2],
     "What should I do to see request options for an unavailable QuickSearch item?",
     "Sign in.", False),
    ("library_interlibrary_loan", [2],
     "Where can I receive requested materials other than articles and book chapters?",
     "At a Rutgers library of my choice or at home.", False),
    ("library_interlibrary_loan", [2],
     "What is the usual EZBorrow turnaround time?",
     "Usually three to five business days.", False),
    ("library_interlibrary_loan", [2],
     "How long is the EZBorrow loan period?",
     "12 weeks.", False),
    ("library_interlibrary_loan", [3],
     "Which service should I try after QuickSearch and EZBorrow cannot find an item?",
     "ILLiad.", False),
    ("library_interlibrary_loan", [3],
     "What are first-time ILLiad users prompted to do?",
     "Register.", False),
    ("library_interlibrary_loan", [4],
     "Why are required textbooks unavailable through interlibrary loan?",
     "They are generally used as course reserves at their home institutions and are not eligible for loan.", True),
    ("library_interlibrary_loan", [4],
     "Where does the library page suggest students purchase or rent textbooks?",
     "The Rutgers University Bookstore.", False),

    ("it_eduroam", [1],
     "Can visitors from eduroam member institutions connect at Rutgers?",
     "Yes.", False),
    ("it_eduroam", [1],
     "What credentials should Rutgers users use when configuring eduroam?",
     "Their NetID and password.", False),
    ("it_eduroam", [1],
     "Who should Rutgers community members contact for eduroam connection problems?",
     "The Help Desk.", False),
    ("it_eduroam", [1],
     "Who should visiting students work with to set up eduroam?",
     "Their home institution.", True),
    ("it_eduroam", [2],
     "Which wireless service is identified for Rutgers community members on campus?",
     "RUWireless Secure.", False),
    ("it_eduroam", [2],
     "Which wireless service is identified for guests?",
     "RUWireless.", False),

    ("it_two_step_login", [1],
     "How many digits are in the Duo identity confirmation code described on the page?",
     "Three.", True),
    ("it_two_step_login", [1],
     "Approximately how long does signing up for two-step login take?",
     "About five minutes.", False),
    ("it_two_step_login", [1],
     "Where are the two-step login help materials available?",
     "The knowledge base within the Rutgers IT Help portal.", False),

    ("it_microsoft_office", [1],
     "Where can I find Office video tutorials?",
     "LinkedIn Learning.", False),
    ("it_microsoft_office", [1],
     "Which listed Office applications are marked PC only?",
     "Publisher and Access.", False),
    ("it_microsoft_office", [1],
     "Can the Office installation allowance include personal equipment?",
     "Yes.", True),
    ("it_microsoft_office", [1, 2],
     "Which storage service does the collected Office page recommend to students?",
     "Google Drive through their ScarletMail accounts.", False),
    ("it_microsoft_office", [2],
     "While what status does the Office license remain active?",
     "While enrolled in or employed at Rutgers.", True),
]

UNSUPPORTED = [
    ("What is the exact graduation application deadline for spring 2027?",
     "New Brunswick", "MS Computer Science",
     "The collected document does not provide that date."),
    ("What is the tuition per credit for MS Computer Science?",
     "New Brunswick", "MS Computer Science",
     "Tuition is not provided."),
    ("What is the minimum GRE score for admission to MS Computer Science?",
     "New Brunswick", "MS Computer Science",
     "Admission test requirements are not provided."),
    ("What is the maximum number of online courses I can take in MSCS?",
     "New Brunswick", "MS Computer Science",
     "The source refers to other policies but does not give this limit."),
    ("What is the deadline for submitting my MSCS essay this semester?",
     "New Brunswick", "MS Computer Science",
     "No specific essay submission deadline is supplied."),
    ("Which professor is available to supervise my thesis this semester?",
     "New Brunswick", "MS Computer Science",
     "Faculty availability is not supplied."),
    ("How many credits are required for an MBA at Rutgers Newark?",
     "Newark", "MBA",
     "The MSCS credit policy does not establish MBA requirements."),
    ("What GPA is required to graduate from the Newark MBA program?",
     "Newark", "MBA",
     "No MBA graduation policy is collected."),
    ("Who approves a history master's thesis at Rutgers Camden?",
     "Camden", "MA History",
     "No history thesis policy is collected."),
    ("What essay sections are required for an MBA at Rutgers Newark?",
     "Newark", "MBA",
     "The MSCS essay format does not establish MBA requirements."),
    ("What is the overdue fine per day for an EZBorrow book?",
     None, None,
     "No overdue fine amount is supplied."),
    ("How many times can I renew an EZBorrow loan?",
     None, None,
     "Renewal limits are not supplied."),
    ("How much does home delivery of an interlibrary loan book cost?",
     None, None,
     "Home delivery is mentioned, but its price is not supplied."),
    ("Can I request an interlibrary loan if I am an alumnus with no other Rutgers affiliation?",
     None, None,
     "The source does not explicitly establish alumni eligibility or ineligibility."),
    ("What exact WiFi settings should I enter to configure eduroam on Windows 11?",
     None, None,
     "The collected page does not contain the detailed configuration steps."),
    ("What download speed does Rutgers guarantee on eduroam?",
     None, None,
     "No guaranteed speed is supplied."),
    ("What are the exact steps to transfer Duo to a replacement phone?",
     None, None,
     "A device guide is mentioned, but its procedural contents are not collected."),
    ("How do I generate a Duo bypass code while traveling abroad?",
     None, None,
     "Travel and bypass help are mentioned without the procedure."),
    ("Which minimum iOS version is required by Duo Mobile?",
     None, None,
     "The page suggests checking compatibility but gives no minimum version."),
    ("How many gigabytes of Google Drive storage does my Rutgers account include?",
     None, None,
     "Google Drive is mentioned, but no storage quota is supplied."),
]

cases = []

for number, (document, indices, question, expected, overlap) in enumerate(
    SUPPORTED, start=1
):
    is_mscs = document == "ms_cs_requirements"
    cases.append({
        "id": f"s{number:02d}",
        "question": question,
        "campus": "New Brunswick" if is_mscs else "Newark",
        "program": "MS Computer Science" if is_mscs else "MBA",
        "answerable": True,
        "relevant_chunk_ids": [
            f"{document}_{index:03d}" for index in indices
        ],
        "expected_answer": expected,
        "development_topic_overlap": overlap,
    })

for number, (question, campus, program, reason) in enumerate(
    UNSUPPORTED, start=1
):
    cases.append({
        "id": f"u{number:02d}",
        "question": question,
        "campus": campus,
        "program": program,
        "answerable": False,
        "relevant_chunk_ids": [],
        "expected_answer": reason,
        "development_topic_overlap": True,
    })

assert len(SUPPORTED) == 40
assert len(UNSUPPORTED) == 20

dataset = {
    "name": "margin_curated_v1",
    "labels_reviewed": False,
    "description": (
        "Source-derived evaluation on five documents. "
        "Not an independent or general-accuracy benchmark. "
        "Topic-overlap flags are provisional and need review."
    ),
    "cases": cases,
}

with OUTPUT.open("x", encoding="utf-8") as handle:
    json.dump(dataset, handle, indent=2, ensure_ascii=False)
    handle.write("\n")

print(f"Created {OUTPUT}")
print("Review question scope, expected answers, and relevant chunk IDs.")
print("Then change labels_reviewed from false to true.")
