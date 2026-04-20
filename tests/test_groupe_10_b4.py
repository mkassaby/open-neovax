from logic.types import Candidate
from modules.groupe_10_b4 import get_score


def make_candidate(peptide_mut, peptide_wt):
    return Candidate(
        candidate_id="TEST",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )


def test_valid_candidate_passes():
    name, value = get_score(make_candidate("SLYNTVATL", "SLYNTIATL"))
    assert name == "B_sanity_check"
    assert value == 1.0


def test_invalid_character_gets_penalty():
    name, value = get_score(make_candidate("SLXNTVATL", "SLYNTIATL"))
    assert name == "B_sanity_check"
    assert value < 0.0


def test_wt_equal_mut_gets_penalty():
    name, value = get_score(make_candidate("SLYNTVATL", "SLYNTVATL"))
    assert name == "B_sanity_check"
    assert value < 0.0


def test_empty_input_gets_strong_penalty():
    name, value = get_score(make_candidate("", "SLYNTIATL"))
    assert name == "B_sanity_check"
    assert value == -10.0


def test_multiple_failures_stack_penalty():
    name, value = get_score(make_candidate("XXXX", "XXXX"))
    assert name == "B_sanity_check"
    assert value <= -20.0


def test_mut_pos_outside_window():
    c = Candidate(
        candidate_id="TEST",
        peptide_wt="GLWDPFNAV",
        peptide_mut="GLWDPFNAV",
        mut_pos_1based=12,
    )
    name, value = get_score(c)
    assert name == "B_sanity_check"
    assert value < 0.0


def test_strong_penalty_for_invalid_chars():
    name, value = get_score(make_candidate("ALXDF*NQV", "ALDDFANQV"))
    assert name == "B_sanity_check"
    assert value <= -10.0
