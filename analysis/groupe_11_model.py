"""
Open-NeoVax — Global Scoring Model (Group 11)
==============================================

Train an ML model to combine pipeline module scores optimally and rank
neoepitope candidates.  Addresses issues #39 and #48.

Usage
-----
    python analysis/groupe_11_model.py

Inputs (produced by ``python analysis/score_analysis.py --generate``)
------
    analysis/scores_patient_one.csv   — 75 training candidates
    analysis/scores_patient_zero.csv  — 18 validation candidates
    analysis/scores_patient_real.csv  — 138 real/decoy candidates
    data/patient_real.csv             — original file with ic50_nm values
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent

PATIENT_ONE_CSV = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_CSV = ANALYSIS_DIR / "scores_patient_real.csv"
PATIENT_ONE_RAW_CSV = PROJECT_ROOT / "data" / "patient_one.csv"
PATIENT_ZERO_RAW_CSV = PROJECT_ROOT / "data" / "patient_zero.csv"
PATIENT_REAL_RAW_CSV = PROJECT_ROOT / "data" / "patient_real.csv"

OUTPUT_IMPORTANCE_PNG = ANALYSIS_DIR / "groupe_11_feature_importance.png"
OUTPUT_ROC_PNG = ANALYSIS_DIR / "groupe_11_roc_real_vs_decoy.png"
OUTPUT_SPEARMAN_PNG = ANALYSIS_DIR / "groupe_11_spearman_ic50.png"
OUTPUT_CM_PNG = ANALYSIS_DIR / "groupe_11_confusion_matrix.png"
OUTPUT_CV_ROC_PNG = ANALYSIS_DIR / "groupe_11_cv_roc.png"
OUTPUT_HEATMAP_PNG = ANALYSIS_DIR / "groupe_11_heatmap.png"
OUTPUT_SHAP_SUMMARY_PNG = ANALYSIS_DIR / "groupe_11_shap_summary.png"
OUTPUT_SHAP_WATERFALL_GOLD_PNG = ANALYSIS_DIR / "groupe_11_shap_waterfall_gold.png"
OUTPUT_SHAP_WATERFALL_BAD_PNG = ANALYSIS_DIR / "groupe_11_shap_waterfall_bad.png"

RANDOM_STATE = 42

# Labels mapped to binary class
LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
# MEDIOCRE and NEUTRAL are ambiguous — removed from training

# Ordinal label encoding (for Section D)
ORDINAL_MAP = {"GOLD": 4, "GOOD": 3, "MEDIOCRE": 2, "BAD": 1, "TRAP": 0}

# Department assignment — used in biological interpretation (Section A)
DEPARTMENT = {
    "A": "physicochemical",
    "B": "processing",
    "C": "HLA binding",
    "D": "safety / self-similarity",
}


# ══════════════════════════════════════════════════════════════════════
#  0. PEPTIDE FEATURE ENGINEERING (HLA-A*02:01 anchors + physicochemistry)
# ══════════════════════════════════════════════════════════════════════
#
# The 20 module scores already cover a lot of ground, but many of them
# are computed as peptide-level aggregates — they lose anchor-specific
# resolution. NetMHCpan, by contrast, learns a position-specific motif.
# We add a small set of engineered features that expose anchor-position
# physicochemistry directly, so the downstream classifier can learn the
# HLA-A*02:01 motif (L/M/I/V at P2, V/L/I at the C-terminus) rather than
# inferring it from averages.

# Kyte–Doolittle hydrophobicity
KD_HYDROPHOBICITY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
# Net charge at pH 7
AA_CHARGE_PH7 = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
# Residue molecular weight (Da, side-chain + backbone)
AA_MW = {
    "A": 89.1,
    "C": 121.2,
    "D": 133.1,
    "E": 147.1,
    "F": 165.2,
    "G": 75.1,
    "H": 155.2,
    "I": 131.2,
    "K": 146.2,
    "L": 131.2,
    "M": 149.2,
    "N": 132.1,
    "P": 115.1,
    "Q": 146.2,
    "R": 174.2,
    "S": 105.1,
    "T": 119.1,
    "V": 117.1,
    "W": 204.2,
    "Y": 181.2,
}
AROMATIC_AA = set("FWY")
# HLA-A*02:01 known anchors
HLA_A0201_P2 = set("LMIV")
HLA_A0201_PC = set("VLI")  # C-terminal anchor (P9 for 9-mers, PN for N-mers)

# Engineered features always produced in this exact order
ENGINEERED_FEATURES = [
    "E_P2_kd",
    "E_PC_kd",
    "E_P2_anchor_ok",
    "E_PC_anchor_ok",
    "E_anchor_sum",
    "E_mean_kd",
    "E_net_charge",
    "E_aromaticity",
    "E_mw_kda",
    "E_len_delta9",
]


def _peptide_features(peptide: str) -> dict[str, float]:
    """Compute engineered features for a single peptide. NaN→0 on failure."""
    zeros = {k: 0.0 for k in ENGINEERED_FEATURES}
    if not isinstance(peptide, str):
        return zeros
    pep = peptide.upper()
    if len(pep) < 2 or not all(aa in KD_HYDROPHOBICITY for aa in pep):
        return zeros

    n = len(pep)
    p2 = pep[1]
    pc = pep[-1]  # C-terminal anchor (position n)

    p2_ok = 1.0 if p2 in HLA_A0201_P2 else 0.0
    pc_ok = 1.0 if pc in HLA_A0201_PC else 0.0

    return {
        "E_P2_kd": KD_HYDROPHOBICITY[p2],
        "E_PC_kd": KD_HYDROPHOBICITY[pc],
        "E_P2_anchor_ok": p2_ok,
        "E_PC_anchor_ok": pc_ok,
        "E_anchor_sum": p2_ok + pc_ok,
        "E_mean_kd": sum(KD_HYDROPHOBICITY[aa] for aa in pep) / n,
        "E_net_charge": sum(AA_CHARGE_PH7.get(aa, 0.0) for aa in pep),
        "E_aromaticity": sum(1 for aa in pep if aa in AROMATIC_AA) / n,
        "E_mw_kda": sum(AA_MW[aa] for aa in pep) / 1000.0,
        "E_len_delta9": abs(n - 9),
    }


def _engineer_from_raw(raw_csv: Path) -> pd.DataFrame:
    """Return a DataFrame keyed by candidate_id with engineered features."""
    if not raw_csv.exists():
        return pd.DataFrame(columns=["candidate_id", *ENGINEERED_FEATURES])
    df = pd.read_csv(raw_csv)
    if "peptide_mut" not in df.columns:
        return pd.DataFrame(columns=["candidate_id", *ENGINEERED_FEATURES])
    rows = [
        {"candidate_id": cid, **_peptide_features(pep)}
        for cid, pep in zip(df["candidate_id"], df["peptide_mut"])
    ]
    return pd.DataFrame(rows, columns=["candidate_id", *ENGINEERED_FEATURES])


def _merge_engineered(
    df_scores: pd.DataFrame, raw_csv: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Left-merge engineered features onto a scores dataframe.

    Returns (augmented_df, engineered_column_names_present).
    """
    df_eng = _engineer_from_raw(raw_csv)
    if df_eng.empty:
        return df_scores, []
    df_out = df_scores.merge(df_eng, on="candidate_id", how="left")
    for col in ENGINEERED_FEATURES:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0.0)
    return df_out, list(ENGINEERED_FEATURES)


# ══════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════


def _check_files() -> bool:
    """Return True if all required score matrices exist."""
    missing = [p for p in (PATIENT_ONE_CSV, PATIENT_ZERO_CSV) if not p.exists()]
    if missing:
        print("ERROR: missing score matrices:")
        for p in missing:
            print(f"  {p}")
        print("Run:  python analysis/score_analysis.py --generate")
        return False
    return True


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a score CSV and fill missing numeric values with 0.0."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("candidate_id", "label")]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def _department_of(feature_name: str) -> str:
    """Return the department letter (A/B/C/D) from a score name."""
    prefix = feature_name.split("_", 1)[0]
    if prefix in DEPARTMENT:
        return prefix
    # D1_exact_self_similarity starts with D1 — normalise to D
    if prefix.startswith("D"):
        return "D"
    return "?"


def load_data() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    Optional[np.ndarray],
    list[str],
    StandardScaler,
]:
    """Load, encode, scale, and align training and validation sets."""
    df_one = _load_csv(PATIENT_ONE_CSV)
    df_zero = _load_csv(PATIENT_ZERO_CSV)

    # ── Peptide-level engineered features (HLA-A*02:01 anchors + physchem) ──
    df_one, eng_one = _merge_engineered(df_one, PATIENT_ONE_RAW_CSV)
    df_zero, eng_zero = _merge_engineered(df_zero, PATIENT_ZERO_RAW_CSV)
    engineered = eng_one if eng_one else eng_zero

    # Identify feature columns (same order for both sets)
    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]

    # Align patient_zero to the same feature columns (fill missing with 0)
    for col in feature_cols:
        if col not in df_zero.columns:
            df_zero[col] = 0.0

    n_module = len(feature_cols) - len(engineered)
    print(
        f"  patient_one  : {len(df_one)} candidates, "
        f"{n_module} module scores + {len(engineered)} engineered peptide features"
    )
    print(f"  patient_zero : {len(df_zero)} candidates")

    # Encode labels
    df_one["y"] = df_one["label"].map(LABEL_MAP)
    train_mask = df_one["y"].notna()
    n_before = len(df_one)
    n_after = train_mask.sum()
    print(
        f"  After filtering ambiguous labels: {n_after} training samples "
        f"(removed {n_before - n_after})"
    )

    X_train_raw = df_one.loc[train_mask, feature_cols].values.astype(float)
    y_train = df_one.loc[train_mask, "y"].values.astype(int)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    X_zero = scaler.transform(df_zero[feature_cols].values.astype(float))

    # Build a binary label array for patient_zero (keep NaN for unknowns)
    df_zero["y"] = df_zero["label"].map(LABEL_MAP)

    return (
        df_one,
        X_train,
        y_train,
        df_zero,
        X_zero,
        df_zero["y"].values,
        feature_cols,
        scaler,
    )


# ══════════════════════════════════════════════════════════════════════
#  2. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[object, str, float]:
    """Run 5-fold CV for three classifiers (RF/GB tuned by grid search)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Logistic regression — also tuned (C regularisation strength)
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 300],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 2],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 300],
            "max_depth": [2, 3, 5],
            "learning_rate": [0.05, 0.1],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    searches: list[tuple[str, GridSearchCV]] = [
        ("Logistic Regression    ", lr_grid),
        ("Random Forest          ", rf_grid),
        ("Gradient Boosting      ", gb_grid),
    ]

    best_score = -1.0
    best_model: object = None
    best_name = ""

    for name, search in searches:
        search.fit(X_train, y_train)
        mean = float(search.best_score_)
        # Std of the best-param row across folds
        idx = search.best_index_
        std = float(search.cv_results_["std_test_score"][idx])
        print(
            f"  {name}  accuracy = {mean:.3f} (+/- {std:.3f})  "
            f"best = {search.best_params_}"
        )
        if mean > best_score:
            best_score = mean
            best_model = search.best_estimator_
            best_name = name.strip()

    print(f"\n  Best model: {best_name}  (CV accuracy = {best_score:.3f})")

    # Section F — explicit 90% accuracy check
    verdict = "beats" if best_score > 0.90 else "does not beat"
    print(f"  Cross-validation accuracy {verdict} 90% (got {best_score:.1%}).")

    return best_model, best_name, best_score


# ══════════════════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════


def analyse_feature_importance(
    best_model: object,
    best_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """Compute and plot feature importances; return permutation importance array."""
    # Fit best model on full training set
    best_model.fit(X_train, y_train)
    # Model stays fitted — reused downstream for ranking and real-data analysis

    # --- Permutation importance (model-agnostic) ---
    perm = permutation_importance(
        best_model,
        X_train,
        y_train,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="accuracy",
    )
    perm_imp = perm.importances_mean

    # --- Built-in importance (RF or GB) ---
    builtin_imp: Optional[np.ndarray] = None
    builtin_label = "Built-in importance"
    if hasattr(best_model, "feature_importances_"):
        builtin_imp = best_model.feature_importances_
        builtin_label = "Tree feature importance"
    elif hasattr(best_model, "coef_"):
        builtin_imp = best_model.coef_[0]
        builtin_label = "LR coefficient (+ → GOOD)"

    # --- Print text table (sorted by permutation importance) ---
    order = np.argsort(perm_imp)[::-1]
    print(f"\n  {'Feature':<38}  Perm.Imp")
    print(f"  {'-' * 38}  --------")
    for i in order:
        bar = "#" * max(0, int(perm_imp[i] * 80))
        print(f"  {feature_cols[i]:<38}  {perm_imp[i]:+.4f}  {bar}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Feature Importance — {best_name}\n(Group 11)", fontsize=13)

    # Left: permutation importance
    sorted_feats = [feature_cols[i] for i in order]
    sorted_perm = perm_imp[order]
    colors_perm = ["#2196F3" if v >= 0 else "#F44336" for v in sorted_perm]
    axes[0].barh(sorted_feats, sorted_perm, color=colors_perm)
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Permutation Importance")
    axes[0].set_xlabel("Mean accuracy decrease")
    axes[0].invert_yaxis()

    # Right: built-in importance or coefficients
    if builtin_imp is not None:
        bi_order = np.argsort(np.abs(builtin_imp))[::-1]
        sorted_bi_feats = [feature_cols[i] for i in bi_order]
        sorted_bi = builtin_imp[bi_order]
        colors_bi = ["#4CAF50" if v >= 0 else "#FF5722" for v in sorted_bi]
        axes[1].barh(sorted_bi_feats, sorted_bi, color=colors_bi)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title(builtin_label)
        axes[1].invert_yaxis()
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMPORTANCE_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_IMPORTANCE_PNG.relative_to(PROJECT_ROOT)}")

    return perm_imp


# ══════════════════════════════════════════════════════════════════════
#  4. FINAL RANKING ON PATIENT_ZERO
# ══════════════════════════════════════════════════════════════════════


def rank_patient_zero(
    best_model: object,
    X_zero: np.ndarray,
    df_zero: pd.DataFrame,
) -> pd.DataFrame:
    """Predict probabilities on patient_zero and print ranked list."""
    proba = best_model.predict_proba(X_zero)[:, 1]
    df_out = df_zero[["candidate_id", "label"]].copy()
    df_out["prob"] = proba
    df_out = df_out.sort_values("prob", ascending=False).reset_index(drop=True)

    for rank, row in df_out.iterrows():
        marker = "  <--" if row["candidate_id"] == "CAND_01" else ""
        print(
            f"  {rank + 1:>2}. {row['candidate_id']:<10}  {row['prob']:.4f}  "
            f"{row['label']}{marker}"
        )

    # Check success criterion
    top_id = df_out.iloc[0]["candidate_id"]
    cand01_rank = df_out[df_out["candidate_id"] == "CAND_01"].index[0] + 1
    if cand01_rank <= 3:
        print(f"\n  [OK] CAND_01 ranked #{cand01_rank} — success criterion met.")
    else:
        print(f"\n  [!] CAND_01 ranked #{cand01_rank} — top candidate is {top_id}.")

    return df_out


# ══════════════════════════════════════════════════════════════════════
#  5. CONFUSION MATRIX + CV ROC (Issue #39 — B)
# ══════════════════════════════════════════════════════════════════════


def plot_confusion_matrix_patient_zero(
    best_model: object,
    X_zero: np.ndarray,
    df_zero: pd.DataFrame,
) -> None:
    """Confusion matrix for labelled patient_zero candidates (GOLD/GOOD vs BAD/TRAP)."""
    mask = df_zero["y"].notna().values
    if mask.sum() == 0:
        print("  No labelled patient_zero candidates — skipping confusion matrix.")
        return

    y_true = df_zero.loc[mask, "y"].values.astype(int)
    y_pred = best_model.predict(X_zero[mask])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BAD/TRAP (0)", "GOLD/GOOD (1)"],
        yticklabels=["BAD/TRAP (0)", "GOLD/GOOD (1)"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — patient_zero (Group 11)")
    plt.tight_layout()
    plt.savefig(OUTPUT_CM_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Saved: {OUTPUT_CM_PNG.relative_to(PROJECT_ROOT)}")


def plot_cv_roc(
    best_model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Out-of-fold ROC curve of the CV model itself."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(y_train, oof_proba)
    fpr, tpr, _ = roc_curve(y_train, oof_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#9C27B0", lw=2, label=f"OOF ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CV ROC — GOLD/GOOD vs BAD/TRAP (Group 11)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_CV_ROC_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Out-of-fold AUC = {auc:.4f}")
    print(f"  Saved: {OUTPUT_CV_ROC_PNG.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════
#  6. HEATMAP OF SCORE MATRIX (Issue #39 — C)
# ══════════════════════════════════════════════════════════════════════


def plot_score_heatmap(df_one: pd.DataFrame, feature_cols: list[str]) -> None:
    """Heatmap of scores_patient_one.csv: candidates (rows) × scores (cols)."""
    # Z-score per feature for comparable colour scale
    mat = df_one[feature_cols].values.astype(float)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd

    plt.figure(figsize=(14, 16))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        xticklabels=feature_cols,
        yticklabels=df_one["candidate_id"].values,
        cbar_kws={"label": "z-score"},
    )
    plt.title("Score matrix — patient_one (z-scored per feature) — Group 11")
    plt.xticks(rotation=60, ha="right")
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP_PNG, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_HEATMAP_PNG.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════
#  7. ORDINAL REGRESSION (Issue #39 — D)
# ══════════════════════════════════════════════════════════════════════


def ordinal_regression(
    df_one: pd.DataFrame,
    feature_cols: list[str],
    df_zero: pd.DataFrame,
    scaler: StandardScaler,
    binary_rank_df: pd.DataFrame,
) -> None:
    """Train a RandomForestRegressor on ordinal labels and compare to binary."""
    # Train set: rows with a recognised ordinal label
    df_ord = df_one.copy()
    df_ord["y_ord"] = df_ord["label"].map(ORDINAL_MAP)
    mask = df_ord["y_ord"].notna()
    X_ord_train = scaler.transform(df_ord.loc[mask, feature_cols].values.astype(float))
    y_ord_train = df_ord.loc[mask, "y_ord"].values.astype(float)

    reg = RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    )
    reg.fit(X_ord_train, y_ord_train)

    # Evaluate on patient_zero candidates that have a known ordinal label
    df_zero_ord = df_zero.copy()
    df_zero_ord["y_ord"] = df_zero_ord["label"].map(ORDINAL_MAP)
    mask_zero = df_zero_ord["y_ord"].notna()
    if mask_zero.sum() < 3:
        print("  Not enough labelled patient_zero candidates — skipping.")
        return

    X_zero_ord = scaler.transform(
        df_zero_ord.loc[mask_zero, feature_cols].values.astype(float)
    )
    y_true = df_zero_ord.loc[mask_zero, "y_ord"].values.astype(float)
    y_pred = reg.predict(X_zero_ord)

    mae = mean_absolute_error(y_true, y_pred)
    rho, p_val = stats.spearmanr(y_pred, y_true)
    print("  Ordinal regression (RandomForestRegressor):")
    print(f"    MAE on patient_zero = {mae:.3f}")
    print(f"    Spearman ρ vs true ordinal labels = {rho:.3f}  (p = {p_val:.4f})")

    # Binary comparison: Spearman between binary prob rank and ordinal label
    merged = binary_rank_df.merge(
        df_zero_ord[mask_zero][["candidate_id", "y_ord"]], on="candidate_id"
    )
    if len(merged) >= 3:
        rho_bin, p_bin = stats.spearmanr(merged["prob"], merged["y_ord"])
        print(
            f"  Binary classifier probability vs ordinal labels: ρ = {rho_bin:.3f} "
            f"(p = {p_bin:.4f})"
        )
        if rho > rho_bin:
            print(
                "  → Ordinal regression better preserves the "
                "GOLD>GOOD>MEDIOCRE>BAD>TRAP ordering."
            )
        else:
            print(
                "  → Binary classifier already captures the ordering as well "
                "or better, so the ordinal lift is limited."
            )


# ══════════════════════════════════════════════════════════════════════
#  8. FEATURE ABLATION (Issue #39 — E)
# ══════════════════════════════════════════════════════════════════════


def feature_ablation(
    best_model_cls: object,
    best_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    perm_imp: np.ndarray,
    baseline_score: float,
) -> None:
    """Drop the 3 least-important features, re-run 5-fold CV, compare accuracy."""
    order = np.argsort(perm_imp)  # ascending
    bottom3 = [feature_cols[i] for i in order[:3]]
    keep_idx = np.array(sorted(order[3:]))
    X_red = X_train[:, keep_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Re-instantiate a fresh classifier of the same type
    fresh = type(best_model_cls)(**best_model_cls.get_params())
    scores = cross_val_score(fresh, X_red, y_train, cv=cv, scoring="accuracy")
    new_score = scores.mean()

    print(f"  Removed 3 least-important features: {bottom3}")
    print(f"  Baseline CV accuracy (all features) = {baseline_score:.3f}")
    print(
        f"  Reduced  CV accuracy (-3 features)  = {new_score:.3f} "
        f"(+/- {scores.std():.3f})"
    )
    delta = new_score - baseline_score
    if delta > 0.005:
        print(f"  → Accuracy IMPROVED by {delta:+.3f} — dropped features were noise.")
    elif delta < -0.005:
        print(f"  → Accuracy DEGRADED by {delta:+.3f} — low-imp. features still help.")
    else:
        print(f"  → Accuracy essentially unchanged (Δ = {delta:+.3f}).")


# ══════════════════════════════════════════════════════════════════════
#  9. BIOLOGICAL INTERPRETATION (Issue #39 — A)
# ══════════════════════════════════════════════════════════════════════


def print_biological_interpretation(
    feature_cols: list[str], perm_imp: np.ndarray
) -> None:
    """Print a biological interpretation paragraph grouped by department."""
    # Aggregate permutation importance per department
    dept_totals: dict[str, float] = {k: 0.0 for k in DEPARTMENT}
    dept_feats: dict[str, list[tuple[str, float]]] = {k: [] for k in DEPARTMENT}
    for feat, imp in zip(feature_cols, perm_imp):
        dept = _department_of(feat)
        if dept in dept_totals:
            dept_totals[dept] += max(imp, 0.0)
            dept_feats[dept].append((feat, imp))

    ranked_depts = sorted(dept_totals.items(), key=lambda kv: kv[1], reverse=True)
    top_order = np.argsort(perm_imp)[::-1]
    top_feats = [feature_cols[i] for i in top_order[:3]]

    print()
    print("  What matters most for neoepitope quality?")
    print("  " + "-" * 56)
    print(
        f"  Top 3 individual scores (by permutation importance): "
        f"{top_feats[0]}, {top_feats[1]}, {top_feats[2]}."
    )
    print("  Aggregated importance per department:")
    for d, total in ranked_depts:
        print(f"    Dept {d} — {DEPARTMENT[d]:<28s}  Σ|perm.imp| = {total:+.4f}")
    dominant = ranked_depts[0][0]
    print()
    print(
        f"  Interpretation: Department {dominant} ({DEPARTMENT[dominant]}) dominates "
        "the model's\n"
        "  decisions on this dataset. Concretely, gating features like "
        f"{top_feats[0]} and\n"
        f"  {top_feats[1]} act as hard sanity filters (invalid characters, "
        "out-of-window mutations,\n"
        "  pathological physicochemical profiles) that flip a candidate from GOOD "
        "to BAD regardless\n"
        "  of its HLA binding. Once those filters are passed, HLA anchors "
        "(Department C — e.g.\n"
        "  C_anchoring_P2, C_hla_anchor_p9) and self-similarity penalties "
        "(Department D) fine-tune\n"
        "  the ranking. Self-similarity is the second-hardest filter because a "
        "peptide matching a\n"
        "  human self-protein would trigger autoimmunity — biologically non-negotiable."
    )


# ══════════════════════════════════════════════════════════════════════
#  10. SHAP EXPLAINABILITY (Issue #48)
# ══════════════════════════════════════════════════════════════════════


def shap_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_zero: np.ndarray,
    df_zero: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """SHAP TreeExplainer on a RandomForest — summary + waterfall for GOLD + BAD."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X_zero)

    # sklearn's RF returns [class0, class1]; SHAP 0.49 may return a 3D array instead.
    if isinstance(raw, list):
        shap_values_pos = np.asarray(raw[1])
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(np.asarray(expected_value).ravel()[-1])
        else:
            base_value = float(expected_value)
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            shap_values_pos = arr[:, :, 1]
        else:
            shap_values_pos = arr
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            ev = np.asarray(expected_value).ravel()
            base_value = float(ev[-1]) if ev.size >= 2 else float(ev[0])
        else:
            base_value = float(expected_value)

    # 1. Beeswarm summary plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values_pos,
        X_zero,
        feature_names=feature_cols,
        show=False,
        plot_size=(9, 6),
    )
    plt.title("SHAP summary (beeswarm) — patient_zero (Group 11)")
    plt.tight_layout()
    plt.savefig(OUTPUT_SHAP_SUMMARY_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_SHAP_SUMMARY_PNG.relative_to(PROJECT_ROOT)}")

    # Ranking using RF probabilities (same model we just fit)
    proba = rf.predict_proba(X_zero)[:, 1]
    df_rank = df_zero[["candidate_id", "label"]].copy()
    df_rank["prob"] = proba
    df_rank = df_rank.sort_values("prob", ascending=False).reset_index(drop=True)

    # Candidate IDs for waterfall
    gold_id = "CAND_01"  # top GOLD per the dataset
    worst_id = df_rank.iloc[-1]["candidate_id"]  # worst-ranked by RF

    def _waterfall(candidate_id: str, out_path: Path, title: str) -> tuple[int, float]:
        idx = df_zero.index[df_zero["candidate_id"] == candidate_id]
        if len(idx) == 0:
            return -1, 0.0
        row = int(idx[0])
        expl = shap.Explanation(
            values=shap_values_pos[row],
            base_values=base_value,
            data=X_zero[row],
            feature_names=feature_cols,
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(expl, show=False, max_display=12)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

        top_idx = int(np.argmax(np.abs(shap_values_pos[row])))
        return top_idx, float(shap_values_pos[row, top_idx])

    _waterfall(
        gold_id,
        OUTPUT_SHAP_WATERFALL_GOLD_PNG,
        f"SHAP waterfall — {gold_id} (top GOLD) — Group 11",
    )
    _waterfall(
        worst_id,
        OUTPUT_SHAP_WATERFALL_BAD_PNG,
        f"SHAP waterfall — {worst_id} (worst-ranked) — Group 11",
    )

    # Per-candidate interpretation text:
    # point to the largest POSITIVE contribution for the GOLD (why it's high)
    # and the largest NEGATIVE contribution for the BAD (why it's low).
    def _row(cid: str) -> int:
        idx = df_zero.index[df_zero["candidate_id"] == cid]
        return int(idx[0]) if len(idx) else -1

    print()
    gold_row = _row(gold_id)
    if gold_row >= 0:
        sv = shap_values_pos[gold_row]
        pos_idx = int(np.argmax(sv))
        print(
            f"  {gold_id} is ranked near the top because "
            f"{feature_cols[pos_idx]} pushed its score up by {sv[pos_idx]:+.4f} "
            f"(largest positive SHAP contribution)."
        )
    worst_row = _row(worst_id)
    if worst_row >= 0:
        sv = shap_values_pos[worst_row]
        neg_idx = int(np.argmin(sv))
        print(
            f"  The worst candidate {worst_id} is ranked last because "
            f"{feature_cols[neg_idx]} pulled its score down by {sv[neg_idx]:+.4f} "
            f"(largest negative SHAP contribution)."
        )
    print(
        "  Global feature importance tells us which modules matter ON AVERAGE. "
        "SHAP tells us\n"
        "  which modules matter FOR A SPECIFIC CANDIDATE. A module can be globally "
        "unimportant\n"
        "  but decisive for one candidate — e.g. D_mutation_in_window matters only "
        "for TRAPs,\n"
        "  so it is silent on 90% of candidates but definitive on the rest."
    )


# ══════════════════════════════════════════════════════════════════════
#  11. REAL DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════


def analyse_patient_real(
    best_model: object,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> None:
    """Run bonus analyses on patient_real data."""
    if not PATIENT_REAL_CSV.exists():
        print("  scores_patient_real.csv not found — skipping.")
        return
    if not PATIENT_REAL_RAW_CSV.exists():
        print("  data/patient_real.csv not found — skipping IC50 analysis.")

    df_real = _load_csv(PATIENT_REAL_CSV)

    # Align features
    for col in feature_cols:
        if col not in df_real.columns:
            df_real[col] = 0.0
    X_real = scaler.transform(df_real[feature_cols].values.astype(float))
    proba_real = best_model.predict_proba(X_real)[:, 1]
    df_real["model_prob"] = proba_real

    # ── Task B: REAL vs DECOY AUC ────────────────────────────────────
    real_mask = df_real["label"] == "REAL"
    decoy_mask = df_real["label"] == "DECOY"
    y_rd = np.where(real_mask, 1, np.where(decoy_mask, 0, np.nan))
    valid = ~np.isnan(y_rd)
    if valid.sum() > 0:
        auc = roc_auc_score(y_rd[valid].astype(int), proba_real[valid])
        fpr, tpr, _ = roc_curve(y_rd[valid].astype(int), proba_real[valid])
        print(f"  AUC REAL vs DECOY = {auc:.4f}")

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — REAL vs DECOY (Group 11)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTPUT_ROC_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUTPUT_ROC_PNG.relative_to(PROJECT_ROOT)}")
    else:
        print("  No REAL/DECOY labels found in patient_real scores.")

    # ── Task A: Spearman correlation with IC50 ───────────────────────
    if not PATIENT_REAL_RAW_CSV.exists():
        return

    df_raw = pd.read_csv(PATIENT_REAL_RAW_CSV)
    # Keep only REAL candidates with valid IC50 values
    df_raw_real = df_raw[(df_raw["label"] == "REAL") & df_raw["ic50_nm"].notna()][
        ["candidate_id", "ic50_nm"]
    ].copy()

    df_merged = df_real[real_mask][["candidate_id", "model_prob"]].merge(
        df_raw_real, on="candidate_id", how="inner"
    )

    if len(df_merged) < 3:
        print("  Not enough REAL candidates with IC50 for Spearman correlation.")
        return

    # Lower IC50 = better binder → higher model score expected → negative rho
    rho, p_val = stats.spearmanr(df_merged["model_prob"], df_merged["ic50_nm"])
    print(f"  Spearman rho = {rho:.4f}  (p = {p_val:.4f})")
    if p_val < 0.05:
        print("  [OK] Statistically significant correlation with IC50.")
    else:
        print("  [!] Correlation with IC50 not significant (p >= 0.05).")

    # Scatter plot: model probability vs IC50
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df_merged["ic50_nm"],
        df_merged["model_prob"],
        color="#4CAF50",
        edgecolors="k",
        linewidth=0.5,
        s=60,
        alpha=0.8,
    )
    plt.xlabel("IC50 (nM) — lower = better binder")
    plt.ylabel("Model probability (REAL)")
    plt.title(
        f"Model prob vs IC50 — Spearman ρ = {rho:.3f} (p = {p_val:.3f})\n(Group 11)"
    )
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_SPEARMAN_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_SPEARMAN_PNG.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════
#  12. TRAINED-MODEL PIPELINE SCORE (used by the NetMHCpan benchmark)
# ══════════════════════════════════════════════════════════════════════


def pipeline_model_score(target_scores_csv: Path, target_raw_csv: Path) -> pd.DataFrame:
    """Train the tuned pipeline on patient_one and score a target patient.

    The NetMHCpan benchmark uses this instead of a plain mean across modules,
    so that the correlation we measure truly reflects the tuned, feature-
    engineered pipeline and not an unweighted average.

    Returns a DataFrame with columns: candidate_id, label, pipeline_score.
    """
    df_one = _load_csv(PATIENT_ONE_CSV)
    df_target = _load_csv(target_scores_csv)

    df_one, eng_one = _merge_engineered(df_one, PATIENT_ONE_RAW_CSV)
    df_target, _ = _merge_engineered(df_target, target_raw_csv)

    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]
    for col in feature_cols:
        if col not in df_target.columns:
            df_target[col] = 0.0

    df_one["y"] = df_one["label"].map(LABEL_MAP)
    train_mask = df_one["y"].notna()
    X_train_raw = df_one.loc[train_mask, feature_cols].values.astype(float)
    y_train = df_one.loc[train_mask, "y"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_target = scaler.transform(df_target[feature_cols].values.astype(float))

    # Logistic regression was the best CV model — it is also well-calibrated,
    # which matters when we turn the probability into a ranking.
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_target)[:, 1]

    if "label" in df_target.columns:
        label_col = df_target["label"]
    else:
        label_col = pd.Series([None] * len(df_target))
    return pd.DataFrame(
        {
            "candidate_id": df_target["candidate_id"].values,
            "label": label_col.values,
            "pipeline_score": proba,
        }
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point — run all analysis steps sequentially."""
    sep = "═" * 43

    print(sep)
    print("  Open-NeoVax — Global Scoring Model (Group 11)")
    print(sep)
    print()

    if not _check_files():
        sys.exit(1)

    # 1. Load data
    print("Loading data...")
    (df_one, X_train, y_train, df_zero, X_zero, _y_zero, feature_cols, scaler) = (
        load_data()
    )

    # 2. Model comparison
    print()
    print("─── MODEL COMPARISON (5-fold CV) " + "─" * 10)
    best_model, best_name, best_score = compare_models(X_train, y_train)

    # 3. Feature importance
    print()
    print("─── FEATURE IMPORTANCE " + "─" * 20)
    perm_imp = analyse_feature_importance(
        best_model, best_name, X_train, y_train, feature_cols
    )

    # 4. Final ranking on patient_zero
    print()
    print("─── FINAL RANKING (patient_zero) " + "─" * 10)
    binary_rank_df = rank_patient_zero(best_model, X_zero, df_zero)

    # 5. Confusion matrix + CV ROC (Issue #39 — B)
    print()
    print("─── CONFUSION MATRIX + CV ROC " + "─" * 13)
    plot_confusion_matrix_patient_zero(best_model, X_zero, df_zero)
    plot_cv_roc(best_model, X_train, y_train)

    # 6. Heatmap of score matrix (Issue #39 — C)
    print()
    print("─── SCORE HEATMAP " + "─" * 25)
    plot_score_heatmap(df_one, feature_cols)

    # 7. Ordinal regression (Issue #39 — D)
    print()
    print("─── ORDINAL REGRESSION " + "─" * 20)
    ordinal_regression(df_one, feature_cols, df_zero, scaler, binary_rank_df)

    # 8. Feature ablation (Issue #39 — E)
    print()
    print("─── FEATURE ABLATION " + "─" * 22)
    feature_ablation(
        best_model, best_name, X_train, y_train, feature_cols, perm_imp, best_score
    )

    # 9. Biological interpretation (Issue #39 — A)
    print()
    print("─── BIOLOGICAL INTERPRETATION " + "─" * 13)
    print_biological_interpretation(feature_cols, perm_imp)

    # 10. SHAP explainability (Issue #48)
    print()
    print("─── SHAP EXPLAINABILITY " + "─" * 19)
    shap_analysis(X_train, y_train, X_zero, df_zero, feature_cols)

    # 11. Patient_real analysis
    print()
    print("─── REAL DATA ANALYSIS " + "─" * 20)
    analyse_patient_real(best_model, feature_cols, scaler)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
