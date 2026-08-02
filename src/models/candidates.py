import os
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from src.data.load_and_validate import load_and_process_data
from src.models.candidates import get_candidate_models
from src.utils.config import load_params

def train_and_evaluate():
    params = load_params()
    threshold = params["prediction"]["threshold"]
    
    # Set up local MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-experimentation")
    
    print("Loading and preprocessing data for training...")
    X_train, X_test, y_train, y_test = load_and_process_data()
    
    # Pass y_train to correctly compute scale_pos_weight for class imbalance
    candidates = get_candidate_models(y_train)
    
    # Ensure local models directory exists for Phase 4 API compatibility
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit and save the feature transformer locally so the API can use it
    from src.features.transformers import CreditRiskFeatureEngineer
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(X_train)
    joblib.dump(transformer, models_dir / "feature_transformer.pkl")
    print("Saved feature_transformer.pkl locally to models/")

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"{name}_training"):
            print(f"\n--- Training {name.upper()} ---")
            model.fit(X_train, y_train)
            
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
                
            y_pred = (y_proba >= threshold).astype(int)
            
            roc_auc = roc_auc_score(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("decision_threshold", threshold)
            
            # Log model with trusted types allowed for skops audit validation
            mlflow.sklearn.log_model(
                model, 
                artifact_path=f"{name}_model",
                skops_trusted_types=[
                    "catboost.core.CatBoostClassifier",
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier"
                ]
            )
            
            # If this is our champion ensemble, save it locally as champion_model.pkl for FastAPI
            if name == "ensemble":
                joblib.dump(model, models_dir / "champion_model.pkl")
                print("Saved champion_model.pkl locally to models/ for FastAPI serving!")
            
            print(f"{name.upper()} Results (Threshold: {threshold}):")
            print(f"  ROC-AUC: {roc_auc:.6f}")
            print(f"  PR-AUC:  {pr_auc:.6f}")
            print(classification_report(y_test, y_pred))

    print("\nModel training, local artifact export, and MLflow logging completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()