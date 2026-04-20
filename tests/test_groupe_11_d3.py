"""Tests for the D3 module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_11_d3 import INVALID_PENALTY, get_score


def _make_candidate(
    wt: str = "AAAAAAAAA", mut: str = "AAAAAAAAV", pos: int = 9
) -> Candidate:
    return Candidate(
        candidate_id="TEST_01", peptide_wt=wt, peptide_mut=mut, mut_pos_1based=pos
    )


def test_returns_tuple_of_two():
    result = get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_not_empty():
    name, _ = get_score(_make_candidate())
    assert name != ""


def test_score_value_finite():
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


# D3-specific edge case tests


def test_central_mutation():
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTIATL", 6))
    assert score == 1.0


def test_anchor_mutation():
    _, anchor_score = get_score(_make_candidate("SLYNTVATL", "SAYNTVATL", 2))
    _, central_score = get_score(_make_candidate("SLYNTVATL", "SLYNTIATL", 6))
    assert anchor_score < central_score


def test_anchor_position():
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTVATA", 9))
    assert score == 0.2


def test_identical_peptides():
    # WT == MUT: no actual mutation
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTVATL", 6))
    assert score == INVALID_PENALTY


def test_position_out_of_range():
    # Position exceeds peptide length
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTIATL", 100))
    assert score == INVALID_PENALTY


def test_different_lengths():
    # WT and MUT have different lengths
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTI", 6))
    assert score == INVALID_PENALTY


def test_multiple_differences_invalid():
    # Two residue changes should be invalid for single-substitution logic
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTIATA", 6))
    assert score == INVALID_PENALTY


def test_declared_position_must_match_actual_difference():
    # Actual difference is at position 6 but declared position is 5
    _, score = get_score(_make_candidate("SLYNTVATL", "SLYNTIATL", 5))
    assert score == INVALID_PENALTY
