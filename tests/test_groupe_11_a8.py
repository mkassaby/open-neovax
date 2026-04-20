"""Tests for the A8 module."""

from __future__ import annotations

import math

import pytest

from logic.types import Candidate
from modules.groupe_11_a8 import get_score


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


# A8-specific edge case tests


def test_aromatic_at_p5_scores_higher():
    # P4-P7 = A F T I -> 0.1, 0.9, 0.2, 0.3
    _, score = get_score(_make_candidate("AAAAAAAAA", "SLMAFTIAV", 5))
    assert score == 0.375


def test_glycine_at_p5_scores_lower():
    # P4-P7 = A G T I -> 0.1, 0.0, 0.2, 0.3
    _, score = get_score(_make_candidate("AAAAAAAAA", "SLMAGTIAV", 5))
    assert score == pytest.approx(0.15)


def test_exposed_mutation_gain_increases_score_vs_non_exposed_change():
    # Same MUT peptide in both candidates; only WT differs.
    # Exposed change at P5 (G->F) should increase A8 vs a non-exposed P2 change.
    _, exposed_gain = get_score(_make_candidate("SLMAGTIAV", "SLMAFTIAV", 5))
    _, non_exposed_change = get_score(_make_candidate("SSMAFTIAV", "SLMAFTIAV", 2))
    assert exposed_gain > non_exposed_change


def test_exposed_mutation_loss_decreases_score_vs_non_exposed_change():
    # Same MUT peptide in both candidates; only WT differs.
    # Exposed change at P5 (F->G) should decrease A8 vs a non-exposed P2 change.
    _, exposed_loss = get_score(_make_candidate("SLMAFTIAV", "SLMAGTIAV", 5))
    _, non_exposed_change = get_score(_make_candidate("SSMAGTIAV", "SLMAGTIAV", 2))
    assert exposed_loss < non_exposed_change


def test_none_wt_returns_base_score_without_crash():
    candidate = Candidate(
        candidate_id="TEST_02",
        peptide_wt=None,  # type: ignore[arg-type]
        peptide_mut="SLMAFTIAV",
        mut_pos_1based=5,
    )

    _, score = get_score(candidate)
    assert score == pytest.approx(0.375)
