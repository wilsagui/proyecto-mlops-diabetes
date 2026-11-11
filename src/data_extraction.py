import os
import pandas as pd
from sklearn.datasets import load_diabetes

def extract_data(output_path: str = "data/raw/diabetes_raw.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    df.to_csv(output_path, index=False)
    print(f"✅ Datos extraídos y guardados en: {output_path}")
    return df

if __name__ == "__main__":
    extract_data()
