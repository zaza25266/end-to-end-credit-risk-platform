import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_params
from src.data.validation import validate_raw_data
from src.features.transformers import CreditRiskFeatureEngineer

def load_and_process_data():
    """
    Loads raw data, validates via Pandera, performs train-test split,
    and applies the feature engineering transformer pipeline.
    """
    params = load_params()
    
    raw_path = params["data"]["raw_path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    target_col = params["project"]["target_column"]

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)

    print("Running Pandera schema validation...")
    validated_df = validate_raw_data(df)

    print("Splitting into features (X) and target (y)...")
    X = validated_df.drop(columns=[target_col])
    y = validated_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Fitting and transforming features using CreditRiskFeatureEngineer...")
    transformer = CreditRiskFeatureEngineer()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    print("Data pipeline preprocessing completed successfully!")
    return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    load_and_process_data()