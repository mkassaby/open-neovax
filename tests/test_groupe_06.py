"""Unit tests for module B1 — groupe_06 (proteasome C-terminal proxy)."""

from __future__ import annotations

import pytest

from logic.types import Candidate
from modules.groupe_06 import NEUTRAL_PENALTY, SCORE_NAME, get_score


def _make_candidate(peptide_mut: str = "SLMAFTIAV") -> Candidate:
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )


# ----------------
# 1. Nominal tests
# ----------------


# Test pour un peptide avec un C-terminus favorisé
# (L hydrophobe), doit retourner un score positif
def test_nominal_favoured_cterm() -> None:
    """Peptide ending with L (hydrophobic) should get a positive score."""
    name, value = get_score(_make_candidate("SLMAFTIAL"))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value > 0.0


# Test pour un peptide avec un C-terminus
# neutre (G), doit retourner un score de 0
def test_nominal_neutral_cterm() -> None:
    """Peptide ending with G should get a neutral score."""
    name, value = get_score(_make_candidate("SLMAFTIAG"))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value == 0.0


# Test pour un peptide avec un C-terminus
# défavorisé (K chargé), doit retourner un score négatif
def test_nominal_disfavoured_cterm() -> None:
    """Peptide ending with K (charged) should get a negative score."""
    name, value = get_score(_make_candidate("SLMAFTIAK"))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value < 0.0


# -----------------------
# 2. Tests de cas limites
# -----------------------


# Test pour un peptide d'une seule acide aminé, devrait être traité sans erreur
def test_single_amino_acid_peptide() -> None:
    """A one-residue peptide should still be handled safely."""
    name, value = get_score(_make_candidate("V"))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value > 0.0


# Test pour un peptide avec des lettres
# minuscules et des espaces, devrait être normalisé correctement
def test_lowercase_and_spaces_are_handled() -> None:
    """Lowercase input with surrounding spaces should be normalized."""
    name, value = get_score(_make_candidate("  slmaftiav  "))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value == 1.0


# Test pour un peptide avec un acide aminé
# non standard (X ou Z), devrait retourner une pénalité
def test_non_standard_cterm_returns_penalty() -> None:
    """Unknown residues such as X or Z should return the documented penalty."""
    for peptide in ("SLMAFTIAX", "SLMAFTIAZ", "SLMAFTIA*"):
        name, value = get_score(_make_candidate(peptide))
        assert name == SCORE_NAME
        assert isinstance(value, float)
        assert value == NEUTRAL_PENALTY


# -------------------------
# 3. Tests de cas invalides
# -------------------------


# Test pour un peptide vide, doit retourner une pénalité
def test_empty_peptide_returns_penalty() -> None:
    name, value = get_score(_make_candidate(""))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value == NEUTRAL_PENALTY


# Test pour un peptide avec valeur None, doit retourner une pénalité
def test_none_peptide_returns_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate("PLACEHOLDER")
    monkeypatch.setattr(candidate, "peptide_mut", None)
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value == NEUTRAL_PENALTY


# Test pour un peptide avec une valeur entière, doit retourner une pénalité
def test_integer_peptide_returns_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate("PLACEHOLDER")
    monkeypatch.setattr(candidate, "peptide_mut", 12345)
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value == NEUTRAL_PENALTY


# -------------------
# 4. Tests de contrat
# -------------------


# Vérification que la fonction retourne toujours un tuple (str, float)
@pytest.mark.parametrize(
    "peptide",
    [
        "SLMAFTIAV",
        "AAAAAAAAAA",
        "GILGFVFTL",
        "FMYSDFHFI",
        "EIYKRWII",
    ],
)
def test_return_type_contract(peptide: str) -> None:
    name, value = get_score(_make_candidate(peptide))
    assert isinstance(name, str)
    assert isinstance(value, float)
    assert name == SCORE_NAME
