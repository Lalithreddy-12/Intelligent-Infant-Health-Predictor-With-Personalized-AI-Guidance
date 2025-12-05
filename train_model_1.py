#Logistic_Regression
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

# Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# Load dataset
# =======================
os.makedirs("models", exist_ok=True)
data_path = os.path.join("models", "sample_input.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Expected dataset at {data_path}. Place your CSV there before training.")

df = pd.read_csv(data_path)

# Drop constant columns (same value everywhere)
df = df.loc[:, (df != df.iloc[0]).any()]

# target
if "mortality" not in df.columns:
    raise ValueError("Dataset must contain a 'mortality' column (binary target).")

X = df.drop("mortality", axis=1)
y = df["mortality"]

print(f"Original class distribution: {dict(pd.Series(y).value_counts())}")

# =======================
# Train/test split (SMOTE AFTER split to avoid leakage)
# =======================
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# apply SMOTE on training set only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
print(f"Balanced class distribution (train after SMOTE): {dict(pd.Series(y_train).value_counts())}")

# =======================
# Preprocessor: simple numeric pipeline (impute + scale)
# If you have categorical features, expand this using ColumnTransformer.
# =======================
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Fit preprocessor on X_train and transform train/test
preprocessor.fit(X_train)
X_train_p = preprocessor.transform(X_train)
X_test_p = preprocessor.transform(X_test)

# =======================
# RandomForest tuning (on preprocessed data)
# =======================
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
rf_grid.fit(X_train_p, y_train)
best_rf = rf_grid.best_estimator_
print(f"\n✅ Best RandomForest params: {rf_grid.best_params_}")

# =======================
# XGBoost tuning (on preprocessed numpy arrays)
# =======================
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1
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
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_p, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"\n✅ Best XGBoost params: {xgb_grid.best_params_}")

# =======================
# LightGBM (safe defaults) - train quickly without grid
# =======================
lgb = LGBMClassifier(
    random_state=42,
    boosting_type="gbdt",
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    min_child_samples=10,
    min_child_weight=1e-3,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)
lgb.fit(X_train_p, y_train)
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
best_cat.fit(X_train_p, y_train)
print("\n✅ CatBoost trained")

# =======================
# Logistic Regression baseline
# =======================
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_p, y_train)

# =======================
# Ensembles: Stacking and Voting (using trained base estimators)
# =======================
stacked = StackingClassifier(
    estimators=[
        ("rf", best_rf),
        ("lgb", best_lgb),
        ("cat", best_cat),
    ],
    final_estimator=XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
    cv=5,
    n_jobs=-1,
    passthrough=False
)
stacked.fit(X_train_p, y_train)

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
voting.fit(X_train_p, y_train)

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

best_model, best_auc = None, 0
for name, mdl in models.items():
    try:
        y_pred = mdl.predict(X_test_p)
        proba = mdl.predict_proba(X_test_p)[:, 1] if hasattr(mdl, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba) if proba is not None else float("nan")
        print(f"\n🔎 {name} Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        if proba is not None and auc > best_auc:
            best_model, best_auc = mdl, auc
    except Exception as e:
        print(f"Evaluation failed for {name}: {e}")

if best_model is None:
    # fallback: pick the stacked if nothing else
    best_model = stacked
    print("No model produced valid probabilities; defaulting to stacked ensemble.")

print(f"\n✅ Selected best model: {best_model.__class__.__name__} with ROC-AUC {best_auc:.4f}")

# =======================
# Fit final pipeline (preprocessor + model) on full training data
# =======================
from sklearn.pipeline import Pipeline as SKPipeline

final_pipeline = SKPipeline([
    ("preproc", preprocessor),
    ("model", best_model)
])

# Fit on the (SMOTE) training data
final_pipeline.fit(X_train, y_train)

# Save pipeline + metadata
joblib.dump({
    "pipeline": final_pipeline,
    "feature_names": list(X.columns),
    "metadata": {
        "created_at": str(pd.Timestamp.now()),
        "best_model": best_model.__class__.__name__,
        "best_auc": float(best_auc) if best_auc is not None else None
    }
}, "models/model.pkl")
print(f"\n✅ Final pipeline saved to models/model.pkl")

# Save dataset snapshot (for reproducibility)
df.to_csv("models/sample_input.csv", index=False)
print("✅ Full dataset snapshot saved to models/sample_input.csv")
