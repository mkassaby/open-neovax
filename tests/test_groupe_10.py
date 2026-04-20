from logic.types import Candidate
from modules.groupe_10 import get_score


def make_candidate(peptide_mut):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )


def test_proline_scores_highest():
    name, value = get_score(make_candidate("PLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == 1.0


def test_glycine_scores_lowest():
    name, value = get_score(make_candidate("GLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == -1.0


def test_leucine_has_intermediate_negative_score():
    name, value = get_score(make_candidate("LLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == -0.5


def test_empty_peptide():
    name, value = get_score(make_candidate(""))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == -1.0


def test_compare_p_vs_g_vs_l():
    _, p_score = get_score(make_candidate("PLMAFTIAV"))
    _, g_score = get_score(make_candidate("GLMAFTIAV"))
    _, l_score = get_score(make_candidate("LLMAFTIAV"))

    assert p_score > l_score > g_score


def test_multiple_distinct_scores_exist():
    scores = {
        get_score(make_candidate("PLMAFTIAV"))[1],
        get_score(make_candidate("KLMAFTIAV"))[1],
        get_score(make_candidate("DLMAFTIAV"))[1],
        get_score(make_candidate("VLMAFTIAV"))[1],
        get_score(make_candidate("LLMAFTIAV"))[1],
        get_score(make_candidate("ALMAFTIAV"))[1],
        get_score(make_candidate("GLMAFTIAV"))[1],
    }
    assert len(scores) >= 5


def test_non_standard_first_aa():
    name, value = get_score(make_candidate("XLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert value == -1.0


def test_too_short_peptide():
    name, value = get_score(make_candidate("PLM"))
    assert name == "B_erap_nterm_proxy"
    assert value == -1.0


def test_too_long_peptide():
    name, value = get_score(make_candidate("PLMAFTIAVAAAA"))
    assert name == "B_erap_nterm_proxy"
    assert value == -1.0
