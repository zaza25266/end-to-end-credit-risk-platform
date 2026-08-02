import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from src.utils.config import load_params

class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom stateful feature engineering transformer for the Credit Risk platform.
    Dynamically loads configuration parameters from params.yaml.
    """
    def __init__(self, n_neighbors: int = None):
        params = load_params()
        # Fallback to params.yaml value if n_neighbors is not explicitly provided
        self.n_neighbors = n_neighbors or params.get("preprocessing", {}).get("knn_imputer_n_neighbors", 5)
        self.knn_imputer = None
        self.median_income_ = None

    def fit(self, X: pd.DataFrame, y=None):
        impute_cols = ["MonthlyIncome", "NumberDependingPersons"]
        available_cols = [col for col in impute_cols if col in X.columns]
        
        if available_cols:
            self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.knn_imputer.fit(X[available_cols])
            
        if "MonthlyIncome" in X.columns:
            self.median_income_ = X["MonthlyIncome"].median()
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Drop redundant index columns if present
        for col in ["Unnamed: 0", "Id"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # 1. Imputation handling via fitted KNN imputer
        impute_cols = ["MonthlyIncome", "NumberDependingPersons"]
        available_cols = [col for col in impute_cols if col in X.columns]
        if self.knn_imputer is not None and available_cols:
            X[available_cols] = self.knn_imputer.transform(X[available_cols])
        
        # Fallback fillna for any remaining NaNs
        if "MonthlyIncome" in X.columns:
            X["MonthlyIncome"] = X["MonthlyIncome"].fillna(self.median_income_)
        if "NumberDependingPersons" in X.columns:
            X["NumberDependingPersons"] = X["NumberDependingPersons"].fillna(0)

        # 2. Derived Feature Engineering (Domain Specific Ratios & Interactions)
        if "MonthlyIncome" in X.columns and "DebtRatio" in X.columns:
            X["EstimatedTotalDebt"] = X["MonthlyIncome"] * X["DebtRatio"]
            
        delinq_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
        existing_delinq = [c for c in delinq_cols if c in X.columns]
        if existing_delinq:
            X["TotalDelinquencies"] = X[existing_delinq].sum(axis=1)

        if "RevolvingUtilizationOfUnsecuredLines" in X.columns and "NumberOfOpenCreditLinesAndLoans" in X.columns:
            X["CreditUtilizationPerLine"] = (
                X["RevolvingUtilizationOfUnsecuredLines"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1)
            )

        if "age" in X.columns:
            X["IsYoungAdult"] = (X["age"] < 30).astype(int)
            X["IsSenior"] = (X["age"] > 65).astype(int)

        return X