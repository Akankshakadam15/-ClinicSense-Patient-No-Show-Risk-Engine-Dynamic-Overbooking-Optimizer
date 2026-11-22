"""
ClinicSense – API Server & MLflow Integration
The dashboard currently uses a hand-coded rule-based probability formula.
This module replaces that with a proper FastAPI server that loads the
real trained + calibrated LightGBM model from Notebook 04.

Also hooks into MLflow to log experiment metrics so future model re-training
is reproducible and trackable.

Run: uvicorn src.api_server:app --reload
  or: python -m src.api_server
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import datetime
import warnings

from src.config import MODELS_DIR, RANDOM_STATE, OUTPUTS_DIR
from src.feature_engineering import MODEL_FEATURES

warnings.filterwarnings("ignore")


# --- MLflow tracking ---
def log_experiment(model, model_name: str, metrics: dict,
                   params: dict = None, features: list = None,
                   experiment_name: str = "ClinicSense-NoShow"):
    """
    Log a training run to MLflow.

    Parameters
    ----------
    model           : fitted sklearn-compatible model
    model_name      : name for this model run
    metrics         : dict of evaluation metrics (accuracy, roc_auc, etc.)
    params          : dict of hyperparameters
    features        : list of feature names used
    experiment_name : MLflow experiment name

    Returns
    -------
    run_id : MLflow run ID (str) or None if MLflow unavailable
    """
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", len(features) if features else 0)

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            # Log feature list
            if features:
                mlflow.log_text(json.dumps(features, indent=2),
                                "feature_list.json")

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model",
                                      registered_model_name=model_name)

            return run.info.run_id

    except ImportError:
        print("[MLOps] mlflow not installed. Run: pip install mlflow")
        return None
    except Exception as e:
        print(f"[MLOps] MLflow error: {e}")
        return None


def log_all_models(models_dict: dict, metrics_list: list,
                   features: list = None):
    """
    Log all models from a comparison run.

    Parameters
    ----------
    models_dict  : {model_name: fitted_model}
    metrics_list : list of metric dicts (from evaluation.compute_metrics)
    features     : list of feature names
    """
    run_ids = {}
    for metrics in metrics_list:
        name = metrics.get("model", "unknown")
        model = models_dict.get(name)
        if model is not None:
            run_id = log_experiment(
                model=model,
                model_name=name,
                metrics=metrics,
                features=features,
            )
            run_ids[name] = run_id
    return run_ids


# --- FastAPI app ---
# followed the official tutorial: https://fastapi.tiangolo.com/tutorial/
def create_app(model_path: str = None):
    """
    Create a FastAPI application for serving no-show predictions.

    Parameters
    ----------
    model_path : path to the joblib model file.
                 Defaults to outputs/models/best_model_calibrated.joblib

    Returns
    -------
    FastAPI app instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
        from typing import List, Optional
    except ImportError:
        print("[MLOps] fastapi not installed. Run: pip install fastapi uvicorn")
        return None

    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model_calibrated.joblib")

    app = FastAPI(
        title="ClinicSense API",
        description="Real-time patient no-show risk prediction API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Model state
    app.state.model = None
    app.state.model_path = model_path
    app.state.model_name = "best_model_calibrated"
    app.state.loaded_at = None

    # ── Pydantic schemas ──────────────────────────────────────────────────────
    class PatientInput(BaseModel):
        age: int = Field(..., ge=0, le=115, description="Patient age")
        gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
        scholarship: int = Field(0, ge=0, le=1)
        hypertension: int = Field(0, ge=0, le=1)
        diabetes: int = Field(0, ge=0, le=1)
        alcoholism: int = Field(0, ge=0, le=1)
        handicap: int = Field(0, ge=0, le=5)
        sms_received: int = Field(0, ge=0, le=1)
        lead_time_days: int = Field(..., ge=0, description="Days from booking to appointment")
        appt_weekday: int = Field(0, ge=0, le=6)
        appt_month: int = Field(1, ge=1, le=12)
        sched_hour: int = Field(8, ge=0, le=23)
        is_monday: int = Field(0, ge=0, le=1)
        is_friday: int = Field(0, ge=0, le=1)
        prior_appointments: int = Field(0, ge=0)
        prior_no_shows: int = Field(0, ge=0)
        prior_no_show_rate: float = Field(0.0, ge=0.0, le=1.0)
        condition_count: int = Field(0, ge=0, le=4)
        same_day_booking: int = Field(0, ge=0, le=1)
        neighbourhood_freq: float = Field(0.01, ge=0.0, le=1.0)
        age_group_enc: int = Field(2, ge=0, le=4)

    class PredictionResponse(BaseModel):
        no_show_probability: float
        risk_level: str
        recommended_action: str

    class BatchInput(BaseModel):
        patients: List[PatientInput]

    class BatchResponse(BaseModel):
        predictions: List[PredictionResponse]
        summary: dict

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_name: str
        loaded_at: Optional[str]
        timestamp: str

    # ── Startup event ─────────────────────────────────────────────────────────
    @app.on_event("startup")
    def load_model():
        try:
            if os.path.exists(app.state.model_path):
                app.state.model = joblib.load(app.state.model_path)
                app.state.loaded_at = datetime.datetime.now().isoformat()
                print(f"[API] Model loaded from {app.state.model_path}")
            else:
                print(f"[API] WARNING: model file not found at {app.state.model_path}")
                # this is fine for the demo – dashboard falls back to rule-based
        except Exception as e:
            print(f"[API] Error loading model: {e}")

    # ── Helper ────────────────────────────────────────────────────────────────
    def _predict_single(patient: PatientInput) -> PredictionResponse:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        features = pd.DataFrame([patient.dict()])
        # Ensure correct column order
        for col in MODEL_FEATURES:
            if col not in features.columns:
                features[col] = 0

        features = features[MODEL_FEATURES]
        prob = float(app.state.model.predict_proba(features)[0, 1])

        if prob >= 0.55:
            risk = "High"
            action = "📞 Phone + SMS reminder"
        elif prob >= 0.30:
            risk = "Medium"
            action = "📱 SMS reminder"
        else:
            risk = "Low"
            action = "✓ No action needed"

        return PredictionResponse(
            no_show_probability=round(prob, 4),
            risk_level=risk,
            recommended_action=action,
        )

    # ── Endpoints ─────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return HealthResponse(
            status="healthy",
            model_loaded=app.state.model is not None,
            model_name=app.state.model_name,
            loaded_at=app.state.loaded_at,
            timestamp=datetime.datetime.now().isoformat(),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientInput):
        return _predict_single(patient)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchInput):
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        predictions = [_predict_single(p) for p in batch.patients]

        high_risk = sum(1 for p in predictions if p.risk_level == "High")
        avg_prob = np.mean([p.no_show_probability for p in predictions])
        total_expected_noshows = sum(p.no_show_probability for p in predictions)

        return BatchResponse(
            predictions=predictions,
            summary={
                "total_patients": len(predictions),
                "high_risk_count": high_risk,
                "avg_no_show_probability": round(avg_prob, 4),
                "expected_total_no_shows": round(total_expected_noshows, 1),
                "recommended_extra_bookings": max(0, round(total_expected_noshows)),
            },
        )

    @app.get("/model/info")
    def model_info():
        return {
            "model_name": app.state.model_name,
            "features": MODEL_FEATURES,
            "n_features": len(MODEL_FEATURES),
            "model_path": app.state.model_path,
            "loaded": app.state.model is not None,
        }

    return app


# --- model export ---
def export_model_metadata(model, model_name: str, metrics: dict,
                          features: list = None,
                          output_dir: str = None) -> str:
    """
    Export model metadata as a JSON file for deployment documentation.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "deployment")
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "exported_at": datetime.datetime.now().isoformat(),
        "features": features or MODEL_FEATURES,
        "n_features": len(features or MODEL_FEATURES),
        "metrics": {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, str))},
        "model_type": type(model).__name__,
        "api_endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
            "info": "GET /model/info",
        },
    }

    path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[MLOps] Metadata exported → {path}")
    return path


# --- CLI runner ---
def run_server(host: str = "0.0.0.0", port: int = 8000,
               model_path: str = None):
    """
    Start the FastAPI server with uvicorn.
    Usage: python -m src.api_server
    """
    try:
        import uvicorn
        app = create_app(model_path)
        if app:
            uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        print("[MLOps] uvicorn not installed. Run: pip install uvicorn")


if __name__ == "__main__":
    run_server()
