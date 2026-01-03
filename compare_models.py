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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compare_models():
    print("==========================================")
    print("      Model Comparison Tournament         ")
    print("==========================================")

    # 1. Load Data
    data_path = "mortality_dataset.csv"
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    # 2. Feature Engineering (Crucial for high accuracy)
    print("Applying medical feature engineering...")
    df["risk_low_bw"] = (df["birth_weight"] < 2.5).astype(int)
    df["risk_very_low_bw"] = (df["birth_weight"] < 1.5).astype(int)
    df["risk_teen_mom"] = (df["maternal_age"] < 18).astype(int)
    df["risk_advanced_mom"] = (df["maternal_age"] > 35).astype(int)
    df["risk_no_visits"] = (df["prenatal_visits"] < 3).astype(int)
    df["risk_poor_nutrition"] = (df["nutrition"] < 40).astype(int)
    df["socio_nut_risk"] = ((df["socioeconomic"] == 0) & (df["nutrition"] < 50)).astype(int)
    df["cumulative_risk_flags"] = (
        df["risk_low_bw"] + df["risk_teen_mom"] + df["risk_no_visits"] + (1 - df["immunized"]) * 2
    )

    X = df.drop("mortality", axis=1)
    y = df["mortality"]

    # 3. Balancing
    print(f"Original class distribution: {dict(y.value_counts())}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Balanced class distribution: {dict(y_res.value_counts())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # 4. Models
    print("\nInitializing models...")
    
    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    
    # XGBoost
    xgb = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    
    # LightGBM
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=42)
    
    # CatBoost (Optimized)
    cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    
    # Logistic
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # Ensembles
    voting = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('cat', cat)],
        voting='soft'
    )

    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "XGBoost": xgb,
        "LightGBM": lgb,
        "CatBoost": cat,
        "VotingEnsemble": voting
    }

    print("\nTraining and Evaluating...")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'ROC-AUC':<10}")
    print("-" * 46)

    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = 0
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{name:<20} | {acc:.4%}   | {auc:.4f}")
            results.append((name, acc))
        except Exception as e:
            print(f"{name:<20} | FAILED: {str(e)}")

    best_name, best_acc = max(results, key=lambda x: x[1])
    print("\n------------------------------------------")
    print(f"🏆 Best Performing Model: {best_name} ({best_acc:.4%})")
    print("------------------------------------------")
    print("NOTE: No models were saved. Run 'train_model.py' to save the production model.")

if __name__ == "__main__":
    compare_models()
