import os
import pandas as pd

def clean_data(input_path: str = "data/raw/diabetes_raw.csv", output_path: str = "data/processed/diabetes_cleaned.csv") -> pd.DataFrame:
    input_df = pd.read_csv(input_path)
    cleaned_df = input_df.copy()
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.drop_duplicates()
    for col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ Datos limpios guardados en: {output_path}")
    return cleaned_df

if __name__ == "__main__":
    clean_data()
