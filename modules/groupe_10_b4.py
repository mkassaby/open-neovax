"""
Module B4 — Sanity checks (minimal biological validity).

Hypothesis: a peptide that violates basic biological constraints cannot
be a valid HLA-I ligand and must be heavily penalized regardless of
other scores.

Computation: check four conditions — length 8-11, valid amino acids,
WT != MUT, and mutation position within the peptide. Each failure
gives -10. All pass = 1.0.

Limitations: does not distinguish types of invalidity, does not verify
whether the mutation is biologically real, does not rank valid peptides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate

SCORE_NAME = "B_sanity_check"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
PENALTY_PER_FAILURE = -10.0


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return a sanity check score for a candidate."""
    try:
        mut = candidate.peptide_mut
        wt = candidate.peptide_wt

        if not mut or not wt:
            return (SCORE_NAME, PENALTY_PER_FAILURE)

        mut = mut.strip().upper()
        wt = wt.strip().upper()

        if not mut or not wt:
            return (SCORE_NAME, PENALTY_PER_FAILURE)

        penalties = 0

        if not (8 <= len(mut) <= 11):
            penalties += 1

        if not all(aa in VALID_AA for aa in mut):
            penalties += 1

        if wt == mut:
            penalties += 1

        mut_pos = candidate.mut_pos_1based
        if not isinstance(mut_pos, int) or mut_pos < 1 or mut_pos > len(mut):
            penalties += 1

        if penalties == 0:
            return (SCORE_NAME, 1.0)

        return (SCORE_NAME, penalties * PENALTY_PER_FAILURE)

    except Exception:
        return (SCORE_NAME, PENALTY_PER_FAILURE)
