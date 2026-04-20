from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS_DIR = Path("analysis")
PATIENT_ONE_PATH = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_PATH = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_PATH = ANALYSIS_DIR / "scores_patient_real.csv"


# =====================
# LOAD & INSPECT
# =====================


def load_scores(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"candidate_id", "label", "target", "target_ordinal", "target_real"}
    return [col for col in df.columns if col not in excluded]


def inspect_dataset(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())


# =====================
# BINARY CLASSIFICATION
# =====================


def encode_labels_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    label_map = {
        "GOLD": 1,
        "GOOD": 1,
        "BAD": 0,
        "TRAP": 0,
    }

    df = df[df["label"].isin(label_map.keys())]
    df["target"] = df["label"].map(label_map)

    return df


def prepare_ml_data(df: pd.DataFrame):
    df = encode_labels_binary(df)

    X = df[get_feature_columns(df)]
    y = df["target"]

    return X, y


def evaluate_logistic_regression(X, y):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    scores = cross_val_score(model, X, y, cv=5)

    print("\n=== LOGISTIC REGRESSION ===")
    print(scores)
    print(f"Mean: {scores.mean():.3f}")


def evaluate_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    scores = cross_val_score(model, X, y, cv=5)

    print("\n=== RANDOM FOREST ===")
    print(scores)
    print(f"Mean: {scores.mean():.3f}")


def compute_feature_importance(X, y):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    features = list(zip(X.columns, importances))
    features.sort(key=lambda x: x[1], reverse=True)

    print("\n=== FEATURE IMPORTANCE ===")
    for name, score in features:
        print(f"{name:30s} {score:.3f}")


def rank_candidates(model, df, feature_cols):
    X = df[feature_cols]

    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["score"] = probs
    df = df.sort_values(by="score", ascending=False)

    print("\n=== RANKING (patient_zero) ===")
    for i, row in enumerate(df.itertuples(), 1):
        marker = " <--" if row.candidate_id == "CAND_01" else ""
        print(f"{i:2d}. {row.candidate_id:10s} {row.score:.3f} {row.label}{marker}")


# =====================
# ORDINAL REGRESSION
# =====================


def encode_labels_ordinal(df):
    df = df.copy()

    label_map = {
        "TRAP": 0,
        "BAD": 1,
        "MEDIOCRE": 2,
        "GOOD": 3,
        "GOLD": 4,
    }

    df = df[df["label"].isin(label_map.keys())]
    df["target_ordinal"] = df["label"].map(label_map)

    return df


def prepare_ml_data_ordinal(df):
    df = encode_labels_ordinal(df)

    X = df[get_feature_columns(df)]
    y = df["target_ordinal"]

    return X, y


def evaluate_regression(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

    print("\n=== REGRESSION (MAE) ===")
    print(f"MAE: {-scores.mean():.3f}")


def rank_regression(model, df, feature_cols):
    X = df[feature_cols]

    preds = model.predict(X)

    df = df.copy()
    df["score"] = preds
    df = df.sort_values(by="score", ascending=False)

    print("\n=== RANKING (regression) ===")
    for i, row in enumerate(df.itertuples(), 1):
        marker = " <--" if row.candidate_id == "CAND_01" else ""
        print(f"{i:2d}. {row.candidate_id:10s} {row.score:.3f} {row.label}{marker}")


# =====================
# REAL VS DECOY
# =====================


def evaluate_real_vs_decoy(model, df, feature_cols):
    df = df.copy()

    label_map = {"REAL": 1, "DECOY": 0}
    df = df[df["label"].isin(label_map)]
    df["target_real"] = df["label"].map(label_map)

    X = df[feature_cols]
    y = df["target_real"]

    probs = model.predict_proba(X)[:, 1]

    # AUC normal
    auc = roc_auc_score(y, probs)

    # AUC inversé
    auc_flipped = roc_auc_score(y, 1 - probs)

    print("\n=== REAL vs DECOY ===")
    print(f"AUC: {auc:.3f}")
    print(f"AUC flipped: {auc_flipped:.3f}")


def evaluate_ic50_correlation(model, scores_df, raw_df, feature_cols):
    """Evaluate Spearman correlation between model scores and IC50."""

    # Merge datasets
    df = scores_df.merge(
        raw_df[["candidate_id", "ic50_nm"]], on="candidate_id", how="inner"
    )

    # Keep only REAL candidates
    df = df[df["label"] == "REAL"]

    X = df[feature_cols]

    # Predict scores
    probs = model.predict_proba(X)[:, 1]

    ic50 = df["ic50_nm"]

    # Spearman correlation
    rho, pvalue = spearmanr(probs, ic50)

    print("\n=== IC50 CORRELATION (REAL only) ===")
    print(f"Spearman rho: {rho:.3f}")
    print(f"p-value: {pvalue:.3e}")


# =====================
# MAIN
# =====================


def main():
    df_one = load_scores(PATIENT_ONE_PATH)
    df_zero = load_scores(PATIENT_ZERO_PATH)
    df_real = load_scores(PATIENT_REAL_PATH)

    inspect_dataset("PATIENT_ONE", df_one)
    inspect_dataset("PATIENT_ZERO", df_zero)
    inspect_dataset("PATIENT_REAL", df_real)

    # ---- Classification ----
    X_train, y_train = prepare_ml_data(df_one)
    feature_cols = X_train.columns.tolist()

    evaluate_logistic_regression(X_train, y_train)
    evaluate_random_forest(X_train, y_train)
    compute_feature_importance(X_train, y_train)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    rank_candidates(model, df_zero, feature_cols)
    evaluate_real_vs_decoy(model, df_real, feature_cols)

    # ---- Regression ----
    X_ord, y_ord = prepare_ml_data_ordinal(df_one)

    evaluate_regression(X_ord, y_ord)

    reg = RandomForestRegressor(n_estimators=200)
    reg.fit(X_ord, y_ord)

    rank_regression(reg, df_zero, feature_cols)

    raw_real_df = pd.read_csv("data/patient_real.csv")

    evaluate_ic50_correlation(model, df_real, raw_real_df, feature_cols)


if __name__ == "__main__":
    main()
