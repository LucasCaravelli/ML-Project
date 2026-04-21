from __future__ import annotations


def count_tokens(text: str) -> int:
    """Return the token count for a string under the project's canonical tokenizer."""
    return len(text.split())
