import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA

df_train = pd.read_csv("analysis/scores_patient_one.csv")
df_test = pd.read_csv("analysis/scores_patient_zero.csv")

# 2. PREPARE DATA

label_map = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

df_train_clean = df_train[df_train["label"].isin(label_map.keys())].copy()

y_train = df_train_clean["label"].map(label_map)

X_train = df_train_clean.drop(columns=["candidate_id", "label"])
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)

candidate_ids = df_test["candidate_id"]
labels_test = df_test["label"]

X_test = df_test.drop(columns=["candidate_id", "label"])
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

X_test = X_test[X_train.columns]

# 3. STANDARDIZE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL COMPARISON

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results[name] = {"model": model, "mean_acc": scores.mean(), "std_acc": scores.std()}

best_name = max(results, key=lambda x: results[x]["mean_acc"])
best_model = results[best_name]["model"]
best_model.fit(X_train_scaled, y_train)

print("=" * 60)
print("MODEL COMPARISON (5-fold cross-validation)")
print("=" * 60)
for name, res in results.items():
    print(f"{name:30s}  accuracy = {res['mean_acc']:.3f} (+/- {res['std_acc']:.3f})")
print(f"\nBest model: {best_name}")

# 5. FEATURE IMPORTANCE

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Méthode 1: Logistic Regression coefficients
print("\n[Method 1] Logistic Regression coefficients (top 10):")
lr_model = models["Logistic Regression"]
lr_model.fit(X_train_scaled, y_train)

coefs = pd.Series(lr_model.coef_[0], index=X_train.columns)
coefs_sorted = coefs.abs().sort_values(ascending=False)

for i, feat in enumerate(coefs_sorted.head(10).index, 1):
    direction = "-> GOOD" if coefs[feat] > 0 else "-> BAD"
    print(f"  {i:2d}. {feat:35s}  {coefs[feat]:+.3f}  {direction}")

# Méthode 2: Random Forest Gini importance
print("\n[Method 2] Random Forest Gini importance (top 10):")
rf_model = models["Random Forest"]
rf_model.fit(X_train_scaled, y_train)

imp_gini = pd.Series(rf_model.feature_importances_, index=X_train.columns)
imp_gini = imp_gini.sort_values(ascending=False)

for i, (feat, imp) in enumerate(imp_gini.head(10).items(), 1):
    print(f"  {i:2d}. {feat:35s}  {imp:.3f}")

# Méthode 3: Permutation importance
print("\n[Method 3] Permutation importance (top 10):")
perm_result = permutation_importance(
    best_model, X_train_scaled, y_train, n_repeats=10, random_state=42
)
imp_perm = pd.Series(perm_result.importances_mean, index=X_train.columns)
imp_perm = imp_perm.sort_values(ascending=False)

for i, (feat, imp) in enumerate(imp_perm.head(10).items(), 1):
    print(f"  {i:2d}. {feat:35s}  {imp:.3f}")


# 6. FINAL RANKING


proba = best_model.predict_proba(X_test_scaled)[:, 1]
ranking = sorted(zip(candidate_ids, proba, labels_test), key=lambda x: -x[1])

with open("analysis/ranking_groupe_05.csv", "w") as f:
    f.write("rank,candidate_id,probability,true_label\n")
    for i, (cid, prob, label) in enumerate(ranking, 1):
        f.write(f"{i},{cid},{prob:.6f},{label}\n")

print("\n" + "=" * 60)
print("FINAL RANKING (patient_zero)")
print("=" * 60)
for i, (cid, prob, label) in enumerate(ranking, 1):
    marker = " <--" if label == "GOLD" else ""
    print(f"  {i:3d}. {cid}  {prob:.3f}  {label}{marker}")
