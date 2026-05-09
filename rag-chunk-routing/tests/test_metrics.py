from __future__ import annotations

import pytest

from rag_cr.metrics import _exact_match, _normalize, score, token_f1


def test_normalize_lowercases():
    assert _normalize("Hello World") == "hello world"


def test_normalize_strips_punctuation():
    assert _normalize("Hello, world!") == "hello world"


def test_normalize_collapses_whitespace():
    assert _normalize("hello   world\n\nfoo") == "hello world foo"


def test_exact_match_identical_strings():
    assert _exact_match("Paris", "Paris") == 1.0


def test_exact_match_different_strings():
    assert _exact_match("Paris", "London") == 0.0


def test_token_f1_identical_strings():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_token_f1_no_overlap():
    assert token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_partial_overlap():
    # pred=[the,cat,sat] gold=[the,cat]
    # common=2, precision=2/3, recall=1.0, F1=4/5
    assert token_f1("the cat sat", "the cat") == pytest.approx(0.8)


def test_token_f1_handles_repeated_tokens():
    # pred=[a,a,b] gold=[a,b,b]
    # common = min(2,1)+min(1,2) = 2
    # precision = 2/3, recall = 2/3, F1 = 2/3
    assert token_f1("a a b", "a b b") == pytest.approx(2 / 3)


def test_token_f1_empty_prediction():
    assert token_f1("", "the cat") == 0.0


def test_score_returns_all_required_fields():
    result = score("Paris", "Paris", ["The capital of France is Paris."])
    assert set(result.keys()) == {"em", "f1", "faithfulness", "cost_tokens"}


def test_score_perfect_match():
    result = score("Paris", "Paris", ["The capital of France is Paris."])
    assert result["em"] == 1.0
    assert result["f1"] == 1.0


def test_score_cost_tokens_sums_passages():
    result = score("Paris", "Paris", ["one two three", "four five"])
    assert result["cost_tokens"] == 5
