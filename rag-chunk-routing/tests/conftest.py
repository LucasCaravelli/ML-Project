from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def sample_corpus_text() -> str:
    return (
        "Alpha beta gamma delta. Epsilon zeta eta theta. "
        "Iota kappa lambda mu. Nu xi omicron pi."
    )
