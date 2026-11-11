# proyecto-mlops-diabetes
Proyecto modulo 4 

# End-to-End MLOps System: Diabetes Progression Prediction

## Project Overview
This repository implements an end-to-end MLOps pipeline for a machine learning regression problem using the Scikit-learn Diabetes Dataset. The goal is to predict disease progression based on patient features, progressing from an exploratory notebook to a fully modular production repository.

## Repository Structure
- `notebooks/`: Contains the exploratory data science notebook (`Proyecto_MLOps.ipynb`).
- `src/`: Modular Python scripts for data engineering and model training.
  - `data_extraction.py`: Extracts raw data.
  - `data_validation.py`: Validates schema, data types, and quality.
  - `data_cleaning.py`: Cleans and formats data.
  - `train.py`: Trains regression models, logs metrics with MLflow, and registers the Champion model.
- `app/`: FastAPI serving application.
  - `main.py`: Endpoints for `/health`, `/model-info`, and `/predict`.
  - `schemas.py`: Pydantic input data validation.
- `.github/workflows/`: CI/CD pipeline automation with GitHub Actions (`ci.yml`).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/wilsagui/proyecto-mlops-diabetes.git](https://github.com/wilsagui/proyecto-mlops-diabetes.git)
   cd proyecto-mlops-diabetes

   pip install -r requirements.txt

   python src/data_extraction.py
python src/data_validation.py
python src/data_cleaning.py
python src/train.py

uvicorn app.main:app --reload

Endpoints
GET /health: Health check endpoint.

GET /model-info: Returns active model name and version.

POST /predict: Generates predictions for input patient data.

Error Handling
422 Unprocessable Entity: Returned for invalid input schemas (handled via Pydantic).

503 Service Unavailable: Returned if the Champion model is not loaded.

500 Internal Server Error: Returned for internal prediction failures.

The GitHub Actions workflow (.github/workflows/ci.yml) triggers on pushes to main or master to automatically execute data extraction, validation, cleaning, model training, and registry steps.
