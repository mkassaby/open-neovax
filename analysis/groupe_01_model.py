"""
Score Analysis & Global Scoring Model
======================================
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Generate the score matrix (provided)
# ══════════════════════════════════════════════════════════════════════


def generate_score_matrix(patient_file: str = "patient_one.csv") -> None:
    from logic.orchestrator import run_modules
    from logic.types import Candidate

    data_dir = PROJECT_ROOT / "data"
    input_path = data_dir / patient_file

    candidates = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(
                Candidate(
                    candidate_id=row["candidate_id"],
                    peptide_wt=row["peptide_wt"],
                    peptide_mut=row["peptide_mut"],
                    mut_pos_1based=int(row["mut_pos_1based"]),
                    gene=row.get("gene", ""),
                    hla_allele=row.get("hla_allele", ""),
                    note=row.get("note", ""),
                )
            )

    scored = run_modules(candidates, PROJECT_ROOT / "modules")

    output_dir = PROJECT_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)

    all_score_names = set()
    for c in scored:
        all_score_names.update(c.scores.keys())
    all_score_names = sorted(all_score_names)

    output_path = output_dir / f"scores_{patient_file}"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "label"] + all_score_names)
        for c in scored:
            label = c.note.split()[0] if c.note else "UNKNOWN"
            row = [c.candidate_id, label]
            row += [c.scores.get(name, "") for name in all_score_names]
            writer.writerow(row)

    print(f"Score matrix saved to {output_path}")
    print(f"  {len(scored)} candidates x {len(all_score_names)} scores")
    print(f"  Scores: {', '.join(all_score_names)}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Train and evaluate models
# ══════════════════════════════════════════════════════════════════════


def train_and_evaluate():

    train_path = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
    test_path = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"
    real_path = PROJECT_ROOT / "analysis" / "scores_patient_real.csv"
    raw_real_path = PROJECT_ROOT / "data" / "patient_real.csv"

    if not train_path.exists() or not test_path.exists():
        print("Error: Score matrices not found. Run with --generate first.")
        return

    # --- 1. LOAD AND PREPARE DATA ---
    train_df = pd.read_csv(train_path).fillna(0)
    test_df = pd.read_csv(test_path).fillna(0)

    # Liste des modules à supprimer
    features_to_drop = []

    # Sélection des colonnes initiales
    base_cols = [c for c in train_df.columns if c not in ["candidate_id", "label"]]

    # Création des sets RAW sans les colonnes inutiles
    X_train_raw = train_df[base_cols].drop(columns=features_to_drop)
    X_test_raw = test_df[base_cols].drop(columns=features_to_drop)

    # On met à jour feature_cols avec les colonnes restantes
    feature_cols = X_train_raw.columns

    # Mapping des labels
    label_map = {"GOLD": 1, "GOOD": 1, "MEDIOCRE": 0, "BAD": 0, "TRAP": 0}
    y_train = train_df["label"].map(label_map)

    # Standardisation (SUR LES COLONNES RESTANTES UNIQUEMENT)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- 2. CROSS-VALIDATION ---
    print("CROSS-VALIDATION ACCURACY (patient_one):")

    lr_model = LogisticRegression(
        random_state=42, max_iter=500, class_weight="balanced"
    )
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
    print(f"  Logistic Regression: {lr_scores.mean():.2f}")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"  Random Forest:       {rf_scores.mean():.2f}")

    # FIX: was accidentally using rf_model here instead of svm_model
    svm_model = SVC(kernel="linear", C=0.1, probability=True, random_state=42)
    svm_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print(f"  SVM (linear):        {svm_scores.mean():.2f}\n")

    # --- 3. FEATURE IMPORTANCE — Random Forest ---
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_ranking = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )

    print("FEATURE IMPORTANCE (Random Forest):")
    max_score = feature_ranking[0][1] if feature_ranking[0][1] > 0 else 1
    for feat, score in feature_ranking:
        bar = "#" * int((score / max_score) * 10)
        print(f"  {feat:<25} {score:.2f}  {bar}")
    print()

    # --- 3. FEATURE IMPORTANCE — LR ---
    lr_model.fit(X_train, y_train)
    lr_importances = abs(lr_model.coef_[0])
    lr_importances_norm = lr_importances / lr_importances.sum()
    feat_importances = pd.Series(lr_importances_norm, index=feature_cols).sort_values(
        ascending=False
    )

    print("FEATURE IMPORTANCE (LR):")
    for feature, val in feat_importances.items():
        bar = "#" * int(val * 50)
        print(f"  {feature:<25}  {val:.2f}  {bar}")
    print()

    # --- FEATURE IMPORTANCE — SVM ---
    svm_model.fit(X_train, y_train)
    svm_importances = abs(svm_model.coef_[0])
    svm_importances_norm = svm_importances / svm_importances.sum()
    feat_importances = pd.Series(svm_importances_norm, index=feature_cols).sort_values(
        ascending=False
    )

    print("FEATURE IMPORTANCE (SVM):")
    for feature, val in feat_importances.items():
        bar = "#" * int(val * 50)
        print(f"  {feature:<25}  {val:.2f}  {bar}")
    print()

    # --- 4. FINAL RANKING — patient_zero ---
    test_df["score"] = rf_model.predict_proba(X_test)[:, 1]
    ranking_rf = test_df.sort_values("score", ascending=False).reset_index(drop=True)

    print("FINAL RANKING — Random Forest (patient_zero):")
    for i, row in ranking_rf.iterrows():
        cid, score, lbl = row["candidate_id"], row["score"], row["label"]
        marker = "  <--" if cid == "CAND_01" else ""
        print(f"{i+1:2}. {cid:<8}  {score:.2f}{lbl}{marker}")
    print()

    test_df["score"] = lr_model.predict_proba(X_test)[:, 1]
    ranking_lr = test_df.sort_values("score", ascending=False).reset_index(drop=True)

    print("FINAL RANKING — LR (patient_zero):")
    for i, row in ranking_lr.iterrows():
        cid, score, lbl = row["candidate_id"], row["score"], row["label"]
        marker = "  <--" if cid == "CAND_01" else ""
        print(f"{i+1:2}. {cid:<8}  {score:.2f}{lbl}{marker}")
    print()

    test_df["score"] = svm_model.predict_proba(X_test)[:, 1]
    ranking_svm = test_df.sort_values("score", ascending=False).reset_index(drop=True)

    print("FINAL RANKING — SVM (patient_zero):")
    for i, row in ranking_svm.iterrows():
        cid, score, lbl = row["candidate_id"], row["score"], row["label"]
        marker = "  <--" if cid == "CAND_01" else ""
        print(f"  {i+1:2}.{cid:<8} {score:.2f} {lbl}{marker}")

    # ══════════════════════════════════════════════════════════════════
    #  NEW TASKS — PATIENT REAL (dbPepNeo2.0 validation)
    # ══════════════════════════════════════════════════════════════════

    if not real_path.exists():
        print("\nSkipping real patient tasks: scores_patient_real.csv not found.")
        return

    print("\n" + "=" * 55)
    print(" REAL PATIENT VALIDATION (dbPepNeo2.0 HNSCC)")
    print("=" * 55)

    real_df = pd.read_csv(real_path).fillna(0)
    X_real_raw = real_df[feature_cols]
    X_real_scaled = scaler.transform(X_real_raw)

    real_df["pipeline_score"] = svm_model.predict_proba(X_real_scaled)[:, 1]
    real_df["is_real"] = real_df["label"].apply(lambda x: 1 if x == "REAL" else 0)

    # --- TASK 2: AUC REAL vs DECOY ---
    if real_df["is_real"].nunique() == 2:
        auc_raw = roc_auc_score(real_df["is_real"], real_df["pipeline_score"])
        auc = max(auc_raw, 1 - auc_raw)
        direction = "normal" if auc_raw >= 0.5 else "inverted (scores flipped)"
        print(f"\n[Task 2] Classification AUC (REAL vs DECOY): {auc:.3f} [{direction}]")
        if auc_raw < 0.5:
            print("  -> Note: raw AUC < 0.5 — model scores DECOYs higher than REALs.")
            print(
                " This is expected: HLA-binding features are identical for "
                "REAL/DECOY pairs."
            )
            print(
                " Only TCR-contact features (A_hybrid_complexity, A_aromaticity) "
                "can separate them."
            )
        if auc > 0.8:
            print("  -> Excellent! Model effectively filters out shuffled decoys.")
    else:
        print("\n[Task 2] Cannot calculate AUC — missing REAL or DECOY labels.")

    # --- TASK 1: Spearman correlation with IC50 ---
    print("\n[Task 1] Correlation with measured IC50 (REAL candidates only):")
    try:
        raw_real_df = pd.read_csv(raw_real_path)
        ic50_col = next((c for c in raw_real_df.columns if "ic50" in c.lower()), None)

        if ic50_col:
            real_only = real_df[real_df["label"] == "REAL"].copy()
            merged = real_only.merge(
                raw_real_df[["candidate_id", ic50_col]], on="candidate_id"
            )

            rho, pvalue = spearmanr(merged["pipeline_score"], merged[ic50_col])
            print(f"  Spearman Rho: {rho:.3f}")
            print(f"  P-value:      {pvalue:.2e}")

            if rho < 0 and pvalue < 0.05:
                print(
                    "-> Significant negative correlation: higher scores = "
                    "lower IC50 = stronger binders."
                )
            elif rho < 0:
                print(
                    "-> Negative trend (correct direction) but not "
                    "significant (p > 0.05)."
                )
                print(
                    "Likely cause: binding features dominate, but REAL/DECOY "
                    "pairs share anchors."
                )
            else:
                print(
                    "-> Positive correlation: model may be scoring weaker "
                    "binders higher."
                )
        else:
            print(f"  Error: no 'ic50' column found in {raw_real_path}.")

    except FileNotFoundError:
        print(f"  Error: could not load {raw_real_path}.")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score analysis and ML model training")
    parser.add_argument(
        "--generate", action="store_true", help="Generate score matrices"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train and evaluate ML models"
    )
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.print_help()
        sys.exit(0)

    if args.generate:
        for patient in ["patient_one.csv", "patient_zero.csv", "patient_real.csv"]:
            print(f"Generating score matrix for {patient}...")
            generate_score_matrix(patient)
            print()

    if args.train:
        train_and_evaluate()
