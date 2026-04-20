import pandas as pd

from analysis.groupe_12_model import (
    encode_labels_binary,
    encode_labels_ordinal,
    get_feature_columns,
    prepare_ml_data,
    prepare_ml_data_ordinal,
)


def test_get_feature_columns_excludes_metadata_and_targets():
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2"],
            "label": ["GOLD", "BAD"],
            "target": [1, 0],
            "target_ordinal": [4, 1],
            "target_real": [1, 0],
            "A_score": [0.1, 0.2],
            "B_score": [0.3, 0.4],
            "C_score": [0.5, 0.6],
        }
    )

    feature_cols = get_feature_columns(df)

    assert feature_cols == ["A_score", "B_score", "C_score"]


def test_encode_labels_binary_filters_and_maps_labels_correctly():
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
            "label": ["GOLD", "GOOD", "BAD", "TRAP", "MEDIOCRE"],
            "A_score": [1, 2, 3, 4, 5],
        }
    )

    result = encode_labels_binary(df)

    assert len(result) == 4
    assert set(result["label"]) == {"GOLD", "GOOD", "BAD", "TRAP"}
    assert result.loc[result["label"] == "GOLD", "target"].iloc[0] == 1
    assert result.loc[result["label"] == "GOOD", "target"].iloc[0] == 1
    assert result.loc[result["label"] == "BAD", "target"].iloc[0] == 0
    assert result.loc[result["label"] == "TRAP", "target"].iloc[0] == 0


def test_encode_labels_ordinal_maps_all_ordered_labels():
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
            "label": ["TRAP", "BAD", "MEDIOCRE", "GOOD", "GOLD"],
            "A_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    result = encode_labels_ordinal(df)

    expected = {
        "TRAP": 0,
        "BAD": 1,
        "MEDIOCRE": 2,
        "GOOD": 3,
        "GOLD": 4,
    }

    assert len(result) == 5
    for label, value in expected.items():
        assert result.loc[result["label"] == label, "target_ordinal"].iloc[0] == value


def test_prepare_ml_data_returns_expected_shapes_and_targets():
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
            "label": ["GOLD", "GOOD", "BAD", "TRAP", "MEDIOCRE"],
            "A_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "B_score": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )

    X, y = prepare_ml_data(df)

    assert X.shape == (4, 2)
    assert list(X.columns) == ["A_score", "B_score"]
    assert y.tolist() == [1, 1, 0, 0]


def test_prepare_ml_data_ordinal_returns_expected_shapes_and_targets():
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
            "label": ["TRAP", "BAD", "MEDIOCRE", "GOOD", "GOLD"],
            "A_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "B_score": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )

    X, y = prepare_ml_data_ordinal(df)

    assert X.shape == (5, 2)
    assert list(X.columns) == ["A_score", "B_score"]
    assert y.tolist() == [0, 1, 2, 3, 4]
