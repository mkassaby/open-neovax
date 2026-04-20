from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------
# CHARGEMENT DES DONNEES
# ----------------------


# Cette fonction charge le fichier CSV contenant les données du patient
def load_score_matrix(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# Normalise le label, c'est à dire le met en majuscules
def normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


# Cette fonction crée un dataset binaire, où
# "GOLD" et "GOOD" sont 1, et "BAD" et "TRAP" sont 0.
def build_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
    GOLD / GOOD -> 1
    BAD / TRAP  -> 0
    Drop MEDIOCRE and other labels.
    """
    label_map = {
        "GOLD": 1,
        "GOOD": 1,
        "BAD": 0,
        "TRAP": 0,
    }

    work = df.copy()

    if "label" not in work.columns:
        raise ValueError("Input score matrix must contain a 'label' column.")

    work["label_str"] = work["label"].map(normalize_label)
    work = work[work["label_str"].isin(label_map.keys())].copy()
    work["y"] = work["label_str"].map(label_map)

    return work


# -------------------------
# # EXTRACTION DES FEATURES
# -------------------------


# Cette fonction récupère les colonnes de
# features numériques dans le DataFrame
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "candidate_id",
        "label",
        "label_str",
        "y",
        "dataset",
        "patient",
        "note",
        "gene",
        "hla_allele",
        "peptide_wt",
        "peptide_mut",
        "mut_pos_1based",
    }

    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


# --------------------
# CREATION DES MODELES
# --------------------


# Fonction qui crée un modèle de régression logistique
def make_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


# Fonction qui crée un modèle Random Forest
def make_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# Fonction qui crée un modèle Gradient Boosting
def make_gradient_boosting() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )


# ------------------
# MODELE COMPARAISON
# ------------------


# Cette fonction évalue les modèles
# en utilisant la validation croisée
def evaluate_models(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, str]:
    """
    Compare models by 5-fold CV and return the best one.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models: dict[str, Pipeline] = {
        "logistic_regression": make_logistic_regression(),
        "random_forest": make_random_forest(),
        "gradient_boosting": make_gradient_boosting(),
    }

    print("\n=== Cross-validation results (accuracy) ===")

    results: dict[str, tuple[float, Pipeline]] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        std_score = scores.std()
        results[name] = (mean_score, model)
        print(f"{name}: mean={mean_score:.4f} std={std_score:.4f}")

    best_name = max(results, key=lambda key: results[key][0])
    best_model = results[best_name][1]

    print(f"\nSelected model: {best_name} (best CV performance)")
    return best_model, best_name


# -----------------------
# IMPORTANCE DES FEATURES
# -----------------------


# Cette fonction affiche l'importance des features
def show_feature_importance(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n=== Feature importance ===")

    logreg_pipeline = make_logistic_regression()
    logreg_pipeline.fit(X, y)
    coef = pd.Series(logreg_pipeline.named_steps["clf"].coef_[0], index=X.columns)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

    print("\nTop Logistic Regression coefficients:")
    print(coef.head(10))

    rf_pipeline = make_random_forest()
    rf_pipeline.fit(X, y)
    importances = pd.Series(
        rf_pipeline.named_steps["clf"].feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print("\nTop Random Forest importances:")
    print(importances.head(10))


# --------------------
# RANKING DES PATIENTS
# --------------------


# Cette fonction génère un classement des candidats pour patient_zero
def rank_patient_zero(
    zero_df: pd.DataFrame,
    feature_cols: list[str],
    fitted_model: Pipeline,
) -> pd.DataFrame:
    missing_cols = [col for col in feature_cols if col not in zero_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in patient_zero: {missing_cols}")

    X_zero = zero_df[feature_cols].copy()

    zero_rank = zero_df.copy()
    zero_rank["pred_proba_good"] = fitted_model.predict_proba(X_zero)[:, 1]

    sort_cols = ["pred_proba_good"]
    ascending = [False]

    if "candidate_id" in zero_rank.columns:
        sort_cols.append("candidate_id")
        ascending.append(True)

    zero_rank = zero_rank.sort_values(sort_cols, ascending=ascending).reset_index(
        drop=True
    )

    keep_cols = [
        col
        for col in ["candidate_id", "label", "pred_proba_good"]
        if col in zero_rank.columns
    ]
    return zero_rank[keep_cols]


# ------
# MAIN
# ------


# fonction principale
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group 06 ML model for Open-NeoVax score matrices"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="analysis/scores_patient_one.csv",
        help="Path to score matrix for patient_one",
    )
    parser.add_argument(
        "--zero_csv",
        type=str,
        default="analysis/scores_patient_zero.csv",
        help="Path to score matrix for patient_zero",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="analysis/patient_zero_ranking_groupe_06.csv",
        help="Output ranking CSV",
    )
    args = parser.parse_args()

    # chargement de donné
    train_df = load_score_matrix(Path(args.train_csv))
    zero_df = load_score_matrix(Path(args.zero_csv))

    # preparation des donné
    train_bin = build_binary_dataset(train_df)
    feature_cols = get_feature_columns(train_bin)

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found in training CSV.")

    # selection de feature
    X = train_bin[feature_cols]
    y = train_bin["y"]

    print("=== Dataset summary ===")
    print(f"Training rows kept for binary task: {len(train_bin)}")
    print(f"Number of features: {len(feature_cols)}")
    print("Features:")
    for col in feature_cols:
        print(f" - {col}")

    # evaluation des modeles
    best_model, best_name = evaluate_models(X, y)
    best_model.fit(X, y)

    show_feature_importance(X, y)

    # affichage de classement
    ranking_df = rank_patient_zero(
        zero_df=zero_df,
        feature_cols=feature_cols,
        fitted_model=best_model,
    )

    print(f"\n=== Final ranking using {best_name} ===")
    print(ranking_df)

    # sauveguard de classement
    output_path = Path(args.output_csv)
    ranking_df.to_csv(output_path, index=False)
    print(f"\nSaved ranking to: {output_path}")


if __name__ == "__main__":
    main()
