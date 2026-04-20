# TODO: Implement synthetic QA generation from chunks, stratified across the three question types defined in the config.

from __future__ import annotations

from .types import QAPair


def generate_qa(
    chunks_path: str,
    target_count: int,
    initial_batch_size: int,
    type_distribution: dict[str, float],
) -> list[QAPair]:
    """Generate a stratified set of synthetic QA pairs from chunks."""
    raise NotImplementedError
