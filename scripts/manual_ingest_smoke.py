"""Manual smoke test: real ingestion against a *running* FastAPI server.

Exercises the actual HTTP surface end-to-end:
    POST /ingest -> poll GET /documents/{id} -> POST /query

Unlike the pytest integration test, this does NOT import your app in-process.
It talks to whatever process is actually running (e.g. `uvicorn api.main:app`
or your Docker container), so it also catches wiring bugs in main.py /
server.py / dependencies.py (get_embedder) and real request/response
serialization -- things an in-process pytest test can't see.

Usage:
    export API_BASE_URL=http://localhost:8000      # default shown
    python manual_ingest_smoke.py path/to/real_document.pdf \
        --type paper \
        --query "What does the paper say about X?" \
        --expect "some substring you know is in the source"

Notes:
  - `--type` must match one of your DocumentType values (paper / book /
    documentation).
  - Status string comparison assumes the API returns lowercase status
    strings (e.g. "ready"). If DocumentStatusResponse serializes the enum
    differently, adjust TERMINAL_STATUSES / the final check below.
  - `cited_passages` shape is assumed to be a list of dicts with a "content"
    key, per HarnessAResponse -- adjust the `query()`/`main()` parsing if
    your actual schema differs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 180
TERMINAL_STATUSES = {"ready", "failed"}


def ingest(base_url: str, file_path: Path, document_type: str, title: str) -> str:
    """POST a document file to the /ingest endpoint and return its document ID.

    Args:
        base_url: Base URL of the running API server.
        file_path: Path to the document file to ingest.
        document_type: One of the supported DocumentType values.
        title: Display title for the document.

    Returns:
        The document ID string from the accepted ingestion job.

    Exits with status 1 if the server does not return HTTP 202
    or omits a valid Location header.
    """
    with file_path.open("rb") as f:
        resp = requests.post(
            f"{base_url}/ingest",
            files={"file": (file_path.name, f)},
            data={"title": title, "document_type": document_type},
            timeout=60,
        )
    if resp.status_code != 202:
        print(f"[FAIL] expected 202, got {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)
    body = resp.json()
    document_id = body["document_id"]
    location = resp.headers.get("location")
    if location is None or urlparse(location).path != f"/documents/{document_id}":
        print(f"[FAIL] Location header missing or invalid: {location!r}")
        sys.exit(1)
    print(
        f"[ingest] 202 accepted, document_id={document_id},"
        f" status={body.get('status')}, location={location}"
    )
    return str(document_id)


def poll_until_terminal(base_url: str, document_id: str) -> dict[str, Any]:
    """Poll GET /documents/{id} until the document reaches a terminal status.

    Args:
        base_url: Base URL of the running API server.
        document_id: The document ID returned by ingest().

    Returns:
        The final document status dict (status, title, etc.).

    Raises:
        TimeoutError: If the document does not reach a terminal status
            within POLL_TIMEOUT_SECONDS.
    """
    start = time.monotonic()
    last_status = None
    while time.monotonic() - start < POLL_TIMEOUT_SECONDS:
        resp = requests.get(f"{base_url}/documents/{document_id}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        status = str(body.get("status", "")).lower()
        if status != last_status:
            elapsed = time.monotonic() - start
            print(f"[poll] t={elapsed:5.1f}s status={status}")
            last_status = status
        if status in TERMINAL_STATUSES:
            return dict(body)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Document {document_id} did not reach a terminal status in {POLL_TIMEOUT_SECONDS}s"
    )


def query(base_url: str, query_text: str) -> dict[str, Any]:
    """Send a query to the /query endpoint and return the response.

    Args:
        base_url: Base URL of the running API server.
        query_text: The question to ask.

    Returns:
        The full query response dict (answer, grounded, cited_passages, etc.).
    """
    resp = requests.post(f"{base_url}/query", json={"query": query_text}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    return dict(body)


def main() -> int:
    """Orchestrate a manual smoke test: ingest, poll, and optionally query.

    Parses CLI arguments, runs the ingest-poll cycle, and if --query is
    provided, sends a follow-up query and validates cited passages against
    the --expect substring.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=Path, help="Path to a real PDF or EPUB")
    parser.add_argument("--type", required=True, choices=["paper", "book", "documentation"])
    parser.add_argument("--title", default=None)
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--query", default=None, help="Question answerable from the doc")
    parser.add_argument("--expect", default=None, help="Substring expected in cited passages")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"error: {args.file} does not exist", file=sys.stderr)
        return 1

    title = args.title or args.file.stem
    started = time.monotonic()

    document_id = ingest(args.base_url, args.file, args.type, title)
    final = poll_until_terminal(args.base_url, document_id)
    elapsed = time.monotonic() - started

    status = str(final.get("status", "")).lower()
    if status != "ready":
        print(f"[FAIL] ingestion ended in status={status!r}: {final}")
        return 1

    print(f"[PASS] ingestion READY in {elapsed:.1f}s")

    if args.query:
        result = query(args.base_url, args.query)
        grounded = result.get("grounded")
        cited = result.get("cited_passages", [])
        print(f"[query] grounded={grounded}, {len(cited)} cited passage(s)")
        print(f"[query] answer: {str(result.get('answer', ''))[:300]}")

        if args.expect:
            hit = any(
                args.expect in (p.get("content", "") if isinstance(p, dict) else str(p))
                for p in cited
            )
            if not grounded or not hit:
                print(f"[FAIL] substring {args.expect!r} not found in grounded cited passages")
                return 1
            print("[PASS] grounded, expected content found in cited passages")

    return 0


if __name__ == "__main__":
    sys.exit(main())
