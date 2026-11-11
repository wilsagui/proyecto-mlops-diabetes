from fastapi import FastAPI, HTTPException, status
import mlflow.sklearn
import pandas as pd
from app.schemas import PatientData

app = FastAPI(title="Diabetes MLOps Serving API", version="1.0.0")

MODEL_NAME = "DiabetesRegressionModel"
MODEL_ALIAS = "champion"
model = None

@app.on_event("startup")
def load_champion_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Modelo Champion cargado exitosamente.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el modelo Champion: {e}")
        model = None

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}

@app.get("/model-info", status_code=status.HTTP_200_OK)
def get_model_info():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service/model not available"
        )
    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "status": "active"
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    try:
        input_df = pd.DataFrame([patient.dict()])
        prediction = model.predict(input_df)[0]
        return {
            "prediction": float(prediction),
            "model": MODEL_NAME,
            "alias": MODEL_ALIAS
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failure: {str(e)}"
        )
