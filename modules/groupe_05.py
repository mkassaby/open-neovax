"""
GROUPE 05 — Open-NeoVax
=============================
Hypothesis: A mutant peptide is a better neoepitope candidate if the mutation
induces a significant physicochemical change relative to the wild-type peptide.
Non-conservative mutations that alter hydrophobicity, charge, or steric bulk
are more likely to be recognized as non-self by CD8+ T lymphocytes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

# ──────────────────────────────────────────────────────────────────────
# Import the Candidate type.
# TYPE_CHECKING is False at runtime → no circular import issues.
# This still lets your editor (VS Code, PyCharm) provide autocompletion.
# ──────────────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    from logic.types import Candidate


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# Path to the data/ directory at the project root.
# Useful if your module needs to load an external file
# (PSSM matrix, self-peptide list, etc.).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


SCORE_NAME = "A_delta_wt_vs_mut"

# Multiple physicochemical scales per amino acid
HYDRO = {  # Kyte-Doolittle
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
CHARGE = {  # at pH 7
    "D": -1,
    "E": -1,
    "K": 1,
    "R": 1,
    "H": 0.5,
}  # all others = 0
VOLUME = {  # Zamyatnin, 1972 (Angstrom^3)
    "G": 60,
    "A": 88,
    "S": 89,
    "C": 108,
    "D": 111,
    "P": 112,
    "N": 114,
    "T": 116,
    "E": 138,
    "V": 140,
    "Q": 143,
    "H": 153,
    "M": 162,
    "I": 166,
    "L": 166,
    "K": 168,
    "R": 173,
    "F": 189,
    "Y": 193,
    "W": 227,
}
# Normalization ranges (max - min for each scale)
HYDRO_RANGE = 9.0  # 4.5 - (-4.5)
CHARGE_RANGE = 2.0  # 1 - (-1)
VOLUME_RANGE = 167.0  # 227 - 60
# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════


def _get_hydrophobicity(aa: str) -> float:
    """Get Kyte-Doolittle hydrophobicity value for an amino acid."""
    return HYDRO.get(aa.upper(), 0.0)


def _get_charge(aa: str) -> float:
    """Get charge at pH 7 for an amino acid."""
    return CHARGE.get(aa.upper(), 0.0)


def _get_volume(aa: str) -> float:
    """Get side-chain volume (Zamyatnin, 1972) for an amino acid.

    Returns 100.0 (average volume) for unknown amino acids.
    """
    return VOLUME.get(aa.upper(), 100.0)


def _delta_at_position(wt_aa: str, mut_aa: str) -> float:
    """
    Calculate normalized Euclidean distance at a single position.

    Returns 0 if amino acids are identical.
    Otherwise computes sqrt((Δhyd)² + (Δcharge)² + (Δvolume)²)
    where each delta is normalized by the scale's range.
    """
    if wt_aa == mut_aa:
        return 0.0

    # Normalized differences
    dh = (_get_hydrophobicity(mut_aa) - _get_hydrophobicity(wt_aa)) / HYDRO_RANGE
    dc = (_get_charge(mut_aa) - _get_charge(wt_aa)) / CHARGE_RANGE
    dv = (_get_volume(mut_aa) - _get_volume(wt_aa)) / VOLUME_RANGE

    return math.sqrt(dh**2 + dc**2 + dv**2)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """
    Compute multi-property Euclidean distance between WT and mutant peptides.

    Hypothesis: Mutations inducing significant physicochemical changes between
    the wild-type and mutant peptide are favored as neoepitope candidates.

    Parameters
    ----------
    candidate : Candidate
        - candidate.peptide_mut : mutated sequence
        - candidate.peptide_wt  : wild-type sequence

    Returns
    -------
    tuple[str, float]
        ("A_delta_wt_vs_mut", score) where:
        - score = 0.0  : identical sequences (no mutation)
        - score > 0.0  : cumulative Euclidean distance across all positions
                         (higher = more radical mutation)
        - score = -1.0 : invalid input (empty, different lengths, exception)

    Limitations
    -----------
    - Does not consider mutation position (anchor vs non-anchor)
    - Cumulative score favors multiple minor mutations over one radical change
    - Some conservative mutations can still be immunogenic
    - This score only measures physicochemical change.
    It doesn't tell if the immune system will actually respond.
    """
    try:
        wt = candidate.peptide_wt.strip().upper()
        mut = candidate.peptide_mut.strip().upper()

        if not wt or not mut or len(wt) != len(mut):
            return (SCORE_NAME, -1.0)

        if wt == mut:
            return (SCORE_NAME, 0.0)

        total_distance = sum(_delta_at_position(w, m) for w, m in zip(wt, mut))

        return (SCORE_NAME, total_distance)

    except Exception:
        return (SCORE_NAME, -1.0)
