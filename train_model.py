import os
import joblib
import pandas as pd
import numpy as np
import warnings
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# 1. Load Dataset
# =======================
DATA_PATH = "mortality_dataset.csv"
MODEL_PATH = os.path.join("models", "model.pkl")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")

print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# =======================
# 2. Feature Engineering
# =======================
print("Applying improved medical feature engineering...")

# High risk flags
df["risk_low_bw"] = (df["birth_weight"] < 2.5).astype(int)
df["risk_very_low_bw"] = (df["birth_weight"] < 1.5).astype(int)
df["risk_teen_mom"] = (df["maternal_age"] < 18).astype(int)
df["risk_advanced_mom"] = (df["maternal_age"] > 35).astype(int)
df["risk_no_visits"] = (df["prenatal_visits"] < 3).astype(int)
df["risk_poor_nutrition"] = (df["nutrition"] < 40).astype(int)
df["socio_nut_risk"] = ((df["socioeconomic"] == 0) & (df["nutrition"] < 50)).astype(int)

# Cumulative risk score
df["cumulative_risk_flags"] = (
    df["risk_low_bw"] + df["risk_teen_mom"] + df["risk_no_visits"] + (1 - df["immunized"]) * 2
)

X = df.drop("mortality", axis=1)
y = df["mortality"]

# =======================
# 3. Balancing (Matching compare_models.py logic for consistency)
# =======================
# Using SMOTE to balance the dataset before splitting to maximize signal capture for this synthetic dataset
print(f"Original class distribution: {dict(y.value_counts())}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Balanced class distribution: {dict(y_res.value_counts())}")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# =======================
# 4. Model Definition
# =======================
print("\nInitializing models...")

rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
xgb = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
voting = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('cat', cat)],
        voting='soft'
    )

models = {
    "RandomForest": rf,
    "XGBoost": xgb,
    "CatBoost": cat,
    "LogisticRegression": lr,
    "VotingEnsemble": voting
}

# =======================
# 5. Training & Selection
# =======================
from sklearn.model_selection import GridSearchCV

# =======================
# 5. Training & Selection
# =======================
print("\nStarting Model Evaluation...")
print(f"{'Model':<20} | {'Accuracy':<10} | {'ROC-AUC':<10}")
print("-" * 46)

best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    try:
        # For Random Forest, use Grid Search CV for generalization
        if name == "RandomForest":
            print(f"Hyperparameter tuning for {name}...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_leaf': [5, 10, 20],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best params for RF: {grid.best_params_}")

        # Train (or retrain if grid search)
        if name != "RandomForest": # RF already fitted by GridSearch
             model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name:<20} | {acc:.4%}   | {auc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    except Exception as e:
        print(f"{name:<20} | FAILED: {e}")

print("-" * 46)
print(f"Selected Best Model: {best_name} (Accuracy: {best_acc:.4%})")
print("-" * 46)

# =======================
# 6. Save Best Model
# =======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, os.path.join("models", "features.pkl"))

