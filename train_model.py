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
# 3. Splitting & Balancing (CORRECTED)
# =======================
# CRITICAL FIX: Split FIRST, then balance ONLY the training set.
# This prevents data leakage and ensures generalized probabilities.
print(f"Original class distribution: {dict(y.value_counts())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# 4. Model Definition
# =======================
print("\nInitializing models...")

# Define SMOTE for pipelines and manual application
smote = SMOTE(random_state=42)

# Models
# Note: For tree-based models, we often don't strictly need SMOTE if we use class_weight,
# but we will use it to be consistent with the user's request for handling the dataset.
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
xgb = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# For Voting, we train on the SMOTE-resampled training set
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
from imblearn.pipeline import Pipeline as ImbPipeline

print("\nStarting Model Evaluation...")
print(f"{'Model':<20} | {'Accuracy':<10} | {'ROC-AUC':<10}")
print("-" * 46)

best_model = None
best_acc = 0
best_name = ""

# Pre-calculate resampled training data for non-pipeline models to save time
print("Resampling training data for standalone models...")
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Balanced training distribution: {dict(y_train_res.value_counts())}")

for name, model in models.items():
    try:
        # For Random Forest, use Grid Search CV with Pipeline to prevent leakage during CV
        if name == "RandomForest":
            print(f"Hyperparameter tuning for {name} with CV=10...")
            
            # Create a pipeline: SMOTE -> Classifier
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', model)
            ])
            
            param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [5, 10], # Reduced search space slightly for speed with 10 folds
                'clf__min_samples_leaf': [5, 10],
                'clf__class_weight': ['balanced', None] 
            }
            
            # CV=10 as requested
            grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train) 
            
            # The best estimator is the pipeline. 
            # We can extract the classifier if needed, but for prediction we should use the pipeline 
            # if we want to apply transformations. However, SMOTE is only for training.
            # So the 'best_estimator_' allows predict() which does NOT run SMOTE (correct behavior).
            model = grid.best_estimator_
            print(f"Best params for RF: {grid.best_params_}")

        else:
            # Train on the pre-resampled data
            model.fit(X_train_res, y_train_res)
        
        # Evaluate on the ORIGINAL X_test (never seen, never resampled)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
             y_prob = model.predict_proba(X_test)[:, 1]
        else:
             y_prob = [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
        
        print(f"{name:<20} | {acc:.4%}   | {auc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    except Exception as e:
        print(f"{name:<20} | FAILED: {e}")
        # traceback.print_exc()

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

