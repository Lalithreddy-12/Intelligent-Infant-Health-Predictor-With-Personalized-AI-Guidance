#CatBoost
import os
import joblib
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Suppress LightGBM/CatBoost warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# Load dataset
# =======================
data_path = os.path.join("models", "sample_input.csv")
df = pd.read_csv(data_path)

# Drop constant columns (same value everywhere → breaks LightGBM)
df = df.loc[:, (df != df.iloc[0]).any()]

X = df.drop("mortality", axis=1)
y = df["mortality"]

# Balance with SMOTE
print(f"Original class distribution: {dict(pd.Series(y).value_counts())}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Balanced class distribution: {dict(pd.Series(y_res).value_counts())}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# =======================
# RandomForest tuning
# =======================
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"\n✅ Best RandomForest params: {rf_grid.best_params_}")

# =======================
# XGBoost tuning
# =======================
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Faster, modern XGB setup
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",   # MUCH faster
    n_jobs=-1             # use all CPU cores
)

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.9],
}

xgb_grid = GridSearchCV(
    xgb,
    xgb_params,
    cv=5,                 
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train.values, y_train.values)   # convert pandas → numpy

best_xgb = xgb_grid.best_estimator_
print(f"\n✅ Best XGBoost params: {xgb_grid.best_params_}")

# =======================
# LightGBM with safer defaults
# =======================
lgb = LGBMClassifier(
    random_state=42,
    boosting_type="gbdt",
    objective="binary",
    n_estimators=500,        # more trees
    learning_rate=0.05,      # smaller LR
    min_child_samples=10,    # loosen restrictions
    min_child_weight=1e-3,
    subsample=0.8,
    colsample_bytree=0.8
)
lgb.fit(X_train, y_train)
best_lgb = lgb
print("\n✅ LightGBM trained with safe defaults")

# =======================
# CatBoost (quick train, silent mode)
# =======================
best_cat = CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.1,
    random_seed=42,
    verbose=0
)
best_cat.fit(X_train, y_train)
print("\n✅ CatBoost trained")

# =======================
# Logistic Regression baseline
# =======================
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# =======================
# Ensembles
# =======================
# Stacked Ensemble with XGB as meta-learner
stacked = StackingClassifier(
    estimators=[
        ("rf", best_rf),
        ("lgb", best_lgb),
        ("cat", best_cat),
    ],
    final_estimator=XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
    cv=5,
    n_jobs=-1,
    stack_method="predict_proba"
)
stacked.fit(X_train, y_train)

# Soft Voting Ensemble
voting = VotingClassifier(
    estimators=[
        ("rf", best_rf),
        ("xgb", best_xgb),
        ("lgb", best_lgb),
        ("cat", best_cat),
        ("logreg", log_reg),
    ],
    voting="soft",
    n_jobs=-1
)
voting.fit(X_train, y_train)

# =======================
# Evaluation
# =======================
models = {
    "RandomForest": best_rf,
    "XGBoost": best_xgb,
    "LightGBM": best_lgb,
    "CatBoost": best_cat,
    "LogisticRegression": log_reg,
    "StackedEnsemble": stacked,
    "VotingEnsemble": voting,
}

best_model, best_acc = None, 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n🔎 {name} Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_acc:
        best_model, best_acc = model, acc

# =======================
# Save best model + dataset
# =======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
print(f"\n✅ Best model ({best_model.__class__.__name__}) saved to models/model.pkl with accuracy {best_acc:.4f}")

df.to_csv("models/sample_input.csv", index=False)
print("✅ Full dataset saved to models/sample_input.csv")
