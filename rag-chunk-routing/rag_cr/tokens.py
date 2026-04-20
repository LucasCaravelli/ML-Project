# TODO: Provide a single tokenizer used for chunk-size budgeting and cost_tokens accounting, so chunking, retrieval, and metrics all count the same way.

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Return the token count for a string under the project's canonical tokenizer."""
    raise NotImplementedError
