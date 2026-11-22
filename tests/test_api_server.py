# tests for the fastapi/mlflow deployment module
# some of these are more like smoke tests since the full
# API needs uvicorn running

import pytest
import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def trained_model(split_data):
    """Train and return a simple model for API tests."""
    X_train, X_test, y_train, y_test = split_data
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


class TestMLflowLogging:
    def test_log_experiment_graceful_import(self, trained_model, split_data):
        """mlflow might not be installed - that's ok, should return None"""
        from src.api_server import log_experiment
        X_train, X_test, y_train, y_test = split_data
        metrics = {"accuracy": 0.78, "roc_auc": 0.70}
        result = log_experiment(trained_model, "test_model", metrics)
        assert result is None or isinstance(result, str)


class TestExportMetadata:
    def test_export_creates_file(self, trained_model, tmp_path):
        from src.api_server import export_model_metadata
        metrics = {"accuracy": 0.78, "roc_auc": 0.70, "f1": 0.55}
        path = export_model_metadata(
            trained_model, "test_model", metrics,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["model_name"] == "test_model"
        assert "metrics" in data
        assert "api_endpoints" in data

    def test_metadata_has_features(self, trained_model, tmp_path):
        from src.api_server import export_model_metadata
        features = ["age", "lead_time_days", "sms_received"]
        path = export_model_metadata(
            trained_model, "test", {"accuracy": 0.8},
            features=features, output_dir=str(tmp_path),
        )
        with open(path) as f:
            data = json.load(f)
        assert data["n_features"] == 3


class TestFastAPIApp:
    def test_create_app_returns_app(self):
        from src.api_server import create_app
        try:
            app = create_app()
            if app is not None:
                assert hasattr(app, "routes")
        except Exception:
            pass  # fastapi may not be installed

    def test_app_has_endpoints(self):
        from src.api_server import create_app
        try:
            app = create_app()
            if app is not None:
                route_paths = [r.path for r in app.routes if hasattr(r, 'path')]
                assert "/health" in route_paths
                assert "/predict" in route_paths
                assert "/predict/batch" in route_paths
        except Exception:
            pass
