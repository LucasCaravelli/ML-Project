# TODO: Implement the experiment loop that runs a given system over the evaluation set and writes predictions, aggregated metrics, and run metadata to a timestamped results directory.

from __future__ import annotations

from pathlib import Path

from .config import Config


def run(system_name: str, qa_path: str | Path, config: Config) -> Path:
    """Run one evaluation system over a QA set and return the results directory."""
    raise NotImplementedError
