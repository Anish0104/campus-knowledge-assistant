import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent
STAGING_ROOT = PROJECT_ROOT / "data" / "staging"

# An explicit source list, not an unrestricted crawler.
SOURCES = [
    {
        "document_id": "it_eduroam",
        "filename": "it_eduroam.txt",
        "source_url": "https://it.rutgers.edu/eduroam/",
        "campus": "University-wide",
        "program": "Not program-specific",
        "topic": "eduroam wireless access",
        "effective_year": None,
    },
    {
        "document_id": "it_two_step_login",
        "filename": "it_two_step_login.txt",
        "source_url": "https://it.rutgers.edu/two-step-login/",
        "campus": "University-wide",
        "program": "Not program-specific",
        "topic": "Two-step login with Duo",
        "effective_year": None,
    },
    {
        "document_id": "it_microsoft_office",
        "filename": "it_microsoft_office.txt",
        "source_url": "https://it.rutgers.edu/microsoft-office/",
        "campus": "University-wide",
        "program": "Not program-specific",
        "topic": "Microsoft Office access",
        "effective_year": None,
    },
]

ALLOWED_HOSTS = {"it.rutgers.edu"}


def validate_url(url: str) -> None:
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError(f"URL is outside the approved source host: {url}")


def fetch_page(client: httpx.Client, url: str) -> httpx.Response:
    """Check every redirect before following it."""
    for _ in range(6):
        validate_url(url)
        response = client.get(url)

        if response.is_redirect:
            location = response.headers.get("location")

            if not location:
                raise ValueError("Redirect did not include a destination.")

            url = urljoin(url, location)
            continue

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "text/html" not in content_type:
            raise ValueError(f"Expected HTML, received: {content_type}")

        return response

    raise ValueError("Too many redirects.")


def extract_content(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find("h1") or soup.find("title")

    if heading is None:
        raise ValueError("Page has no title.")

    title = " ".join(heading.get_text(" ", strip=True).split())

    root = None
    selected_selector = ""

    for selector in (
        "main",
        '[role="main"]',
        "article",
        "#main-content",
        "#content",
    ):
        candidate = soup.select_one(selector)

        if candidate is not None:
            root = candidate
            selected_selector = selector
            break

    if root is None:
        raise ValueError("Could not identify the main page content.")

    # Save introductory descriptions even when they are outside main.
    descriptions = [
        " ".join(element.get_text(" ", strip=True).split())
        for element in soup.select(".page-description")
    ]

    # Remove the Duo FAQ link collection.
    # These links contain questions, not the answers on linked pages.
    for element in list(root.find_all(["h2", "h3"])):
        heading_text = " ".join(
            element.get_text(" ", strip=True).split()
        ).casefold()

        if heading_text == "top faqs for two-step login":
            section = element.find_parent("section")

            if section is not None and root in section.parents:
                section.decompose()
            else:
                raise ValueError(
                    "Found Duo FAQ links but could not safely "
                    "identify their containing section."
                )

    # Remove navigation, embedded media, and news listings.
    for selector in (
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "form",
        "button",
        "iframe",
        "svg",
        '[role="navigation"]',
        ".breadcrumb",
        ".breadcrumbs",
        ".cc--topic-listing",
    ):
        for element in list(root.select(selector)):
            element.decompose()

    block_names = {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "dt", "dd", "tr",
    }

    # Include description divs alongside ordinary text blocks.
    selector = ", ".join(sorted(block_names)) + ", .page-description"
    elements = root.select(selector)
    selected_ids = {id(element) for element in elements}

    blocks = [title]
    seen = {title}

    def add_block(text: str) -> None:
        text = " ".join(text.split())

        if text and text not in seen:
            blocks.append(text)
            seen.add(text)

    for description in descriptions:
        add_block(description)

    for element in elements:
        # Avoid extracting a paragraph twice when it is inside a list item.
        if any(
            id(parent) in selected_ids
            for parent in element.parents
            if parent is not root
        ):
            continue

        add_block(element.get_text(" ", strip=True))

    text = "\n\n".join(blocks)

    if len(text.split()) < 40:
        raise ValueError("Extracted content is unexpectedly short.")

    if len(text) > 100_000:
        raise ValueError("Extracted content is unexpectedly large.")

    return title, text + "\n", selected_selector


def find_previous_hash(document_id: str, current_run: Path) -> str | None:
    previous_manifests = sorted(
        STAGING_ROOT.glob("*/manifest.json"),
        reverse=True,
    )

    for manifest_path in previous_manifests:
        if manifest_path.parent == current_run:
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for document in manifest.get("documents", []):
            if document["document_id"] == document_id:
                return document.get("content_sha256")

    return None


def main() -> int:
    started_at = datetime.now(timezone.utc)
    run_name = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = STAGING_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    documents = []
    errors = []

    with httpx.Client(
        timeout=30.0,
        follow_redirects=False,
        headers={
            "User-Agent": "MarginStudentProject/0.1",
            "Accept": "text/html",
        },
    ) as client:
        for index, source in enumerate(SOURCES):
            if index:
                time.sleep(1)

            document_id = source["document_id"]
            print(f"Collecting: {document_id}", flush=True)

            try:
                response = fetch_page(client, source["source_url"])

                title, text, selector = extract_content(response.text)

                content_hash = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()

                previous_hash = find_previous_hash(document_id, run_dir)

                if previous_hash is None:
                    change_status = "new"
                elif previous_hash == content_hash:
                    change_status = "unchanged"
                else:
                    change_status = "changed"

                text_path = run_dir / source["filename"]
                text_path.write_text(text, encoding="utf-8")

                # Keep the downloaded HTML for extraction troubleshooting.
                html_path = run_dir / f"{document_id}.html"
                html_path.write_text(response.text, encoding="utf-8")

                documents.append({
                    **source,
                    "title": title,
                    "source_url": str(response.url),
                    "requested_url": source["source_url"],
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": content_hash,
                    "word_count": len(text.split()),
                    "extraction_selector": selector,
                    "change_status": change_status,
                    "review_status": "pending",
                })

                print(
                    f"  SAVED: {len(text.split())} words "
                    f"({change_status})"
                )
                print(f"  Title: {title}")
                print(f"  Preview: {text[:180]!r}\n")

            except (httpx.HTTPError, ValueError, OSError) as error:
                errors.append({
                    "document_id": document_id,
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"  FAILED: {error}\n")

    manifest = {
        "created_at": started_at.isoformat(),
        "documents": documents,
        "errors": errors,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Collected: {len(documents)}/{len(SOURCES)}")
    print(f"Failures: {len(errors)}")
    print(f"Review folder: {run_dir.relative_to(PROJECT_ROOT)}")
    print("The active catalog and search index were not changed.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())