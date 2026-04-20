"""
D3 — Mutation within window (Group 11)
=======================================

Validates that the mutation exists and is consistent in the peptide.
Checks: WT and MUT are different, position is in range,
and the residue at the position actually differs.

Invalid candidates get a strong penalty.
Valid candidates get a score based on the mutation position.
"""

from __future__ import annotations

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

# Name of the score returned by this module.
# IMPORTANT: this name must be unique across all modules!
# Convention: <department>_<concept>[_detail]
SCORE_NAME = "D_mutation_in_window"

# Scoring policy: neutral value for valid candidates, strong penalty for
# inconsistent mutation data.
NEUTRAL_SCORE = 0.0
INVALID_PENALTY = -1000.0


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _validate_mutation_in_window(
    peptide_wt: str, peptide_mut: str, mut_pos_1based: int
) -> bool:
    """Return True when WT/MUT sequences and declared position are consistent."""

    # WT must be a non-empty string.
    if not isinstance(peptide_wt, str) or not peptide_wt:
        return False

    # MUT must be a non-empty string.
    if not isinstance(peptide_mut, str) or not peptide_mut:
        return False

    # WT and MUT must describe the same peptide window length.
    if len(peptide_wt) != len(peptide_mut):
        return False

    # Mutation position must be an integer.
    if not isinstance(mut_pos_1based, int):
        return False

    # Position is 1-based and must fall inside the peptide.
    if mut_pos_1based < 1 or mut_pos_1based > len(peptide_mut):
        return False

    # For single-substitution candidates, WT and MUT must differ at exactly
    # one position.
    diff_positions = [
        i
        for i, (aa_wt, aa_mut) in enumerate(zip(peptide_wt, peptide_mut))
        if aa_wt != aa_mut
    ]
    if len(diff_positions) != 1:
        return False

    # Declared mutation position must match the actual unique difference.
    return diff_positions[0] == (mut_pos_1based - 1)


def _position_relevance(mut_pos_1based: int, pep_len: int) -> float:
    """Return a relevance weight based on mutation position in the peptide."""

    if mut_pos_1based in (2, pep_len):
        return 0.2

    if 3 <= mut_pos_1based <= pep_len - 1:
        return 1.0

    return 0.5


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the score for a given candidate.

    This is THE ONLY FUNCTION that the pipeline calls.
    It must ALWAYS return a tuple (str, float):
      - str  : the unique name of your score
      - float : the computed value (not NaN, not inf)

    Parameters
    ----------
    candidate : Candidate
        Object containing neo-epitope information:
        - candidate.peptide_mut  (str)  : mutated sequence
        - candidate.peptide_wt   (str)  : wild-type sequence
        - candidate.mut_pos_1based (int): mutation position
        - candidate.gene         (str)  : source gene
        - candidate.hla_allele   (str)  : target HLA allele

    Returns
    -------
    tuple[str, float]
        (score_name, score_value)
    """
    # 1. Compute the score using your logic
    is_valid = _validate_mutation_in_window(
        candidate.peptide_wt, candidate.peptide_mut, candidate.mut_pos_1based
    )
    if not is_valid:
        score_value = INVALID_PENALTY
    else:
        score_value = _position_relevance(
            candidate.mut_pos_1based, len(candidate.peptide_mut)
        )

    # 2. Return the result in the expected format
    return (SCORE_NAME, score_value)
