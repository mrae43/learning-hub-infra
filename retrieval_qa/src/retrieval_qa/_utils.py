"""Internal shared utilities for the retrieval_qa package."""

from hashlib import sha256


def _sha256(text: str) -> str:
    return sha256(text.encode()).hexdigest()
