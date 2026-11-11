import pandas as pd
import numpy as np

def validate_data(dataframe: pd.DataFrame) -> bool:
    assert 'target' in dataframe.columns, "Target column missing!"
    required_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    for col in required_features:
        assert col in dataframe.columns, f"Feature {col} is missing!"
    assert dataframe.select_dtypes(include=[np.number]).shape[1] == dataframe.shape[1], "Non-numeric data detected!"
    assert dataframe.isnull().sum().sum() == 0, "Critical missing values found!"
    assert len(dataframe) > 0, "Dataset is empty!"
    print("✅ Data Validation Passed!")
    return True

if __name__ == "__main__":
    df = pd.read_csv("data/raw/diabetes_raw.csv")
    validate_data(df)
