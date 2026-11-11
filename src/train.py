import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_select_champion():
    mlflow.set_experiment("Diabetes_Progression_Experiment")
    
    df = pd.read_csv("data/processed/diabetes_cleaned.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    
    best_rmse = float("inf")
    best_model_name = None
    best_run_id = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            print(f"Model: {name} | RMSE: {rmse:.2f} | R2: {r2:.2f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_run_id = run.info.run_id

    # Quality Gate (RMSE < 56.0) y Asignación de Champion
    QUALITY_GATE_RMSE = 56.0
    model_registry_name = "DiabetesRegressionModel"
    
    if best_rmse < QUALITY_GATE_RMSE:
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, model_registry_name)
        
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_registry_name,
            alias="champion",
            version=mv.version
        )
        print(f"🏆 Champion asignado: {best_model_name} (Versión {mv.version}) con RMSE de {best_rmse:.2f}")
    else:
        print("❌ Ningún modelo superó el Quality Gate.")

if __name__ == "__main__":
    train_and_select_champion()
