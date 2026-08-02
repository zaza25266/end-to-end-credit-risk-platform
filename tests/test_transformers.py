import pandas as pd
from src.features.transformers import CreditRiskFeatureEngineer

def test_feature_engineer_transform():
    df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.4,
        "MonthlyIncome": None,  # Test imputation handling
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTimes90DaysLate": 1,
        "NumberDependingPersons": None
    }])
    
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)
    transformed = transformer.transform(df)
    
    # Check that new features are engineered
    assert "EstimatedTotalDebt" in transformed.columns
    assert "TotalDelinquencies" in transformed.columns
    assert "CreditUtilizationPerLine" in transformed.columns
    assert "IsYoungAdult" in transformed.columns
    assert "IsSenior" in transformed.columns
    
    # Check that missing income was imputed safely
    assert not transformed["MonthlyIncome"].isnull().any()
    assert transformed["TotalDelinquencies"].iloc[0] == 3