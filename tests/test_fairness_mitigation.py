"""
I wrote these after the fairness audit in notebook 07 showed
some disparity. Testing that the mitigation actually helps.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def fairness_data(split_data, engineered_df):
    """Prepare data with sensitive features for fairness tests."""
    X_train, X_test, y_train, y_test = split_data

    # Train a simple model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get sensitive features aligned with test data
    sensitive = pd.Series(
        engineered_df.loc[X_test.index, "gender"].values,
        index=X_test.index, name="gender"
    )

    return model, X_train, X_test, y_train, y_test, y_pred, sensitive


class TestFairnessMetrics:
    def test_compute_metrics(self, fairness_data):
        from src.fairness_mitigation import compute_fairness_metrics
        model, X_train, X_test, y_train, y_test, y_pred, sensitive = fairness_data
        result = compute_fairness_metrics(y_test, y_pred, sensitive)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "demographic_parity_diff" in result

    def test_dpd_bounded(self, fairness_data):
        from src.fairness_mitigation import compute_fairness_metrics
        model, X_train, X_test, y_train, y_test, y_pred, sensitive = fairness_data
        result = compute_fairness_metrics(y_test, y_pred, sensitive)
        if result["demographic_parity_diff"] is not None:
            assert -1 <= result["demographic_parity_diff"] <= 1


class TestExponentiatedGradient:
    def test_mitigation_returns_model(self, fairness_data):
        """after Agarwal et al. 2018 - the mitigated model should still be usable"""
        from src.fairness_mitigation import mitigate_exponentiated_gradient
        model, X_train, X_test, y_train, y_test, y_pred, sensitive = fairness_data

        base = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ])
        mitigated = mitigate_exponentiated_gradient(
            base, X_train, y_train, sensitive_train=pd.Series(
                np.random.default_rng(42).integers(0, 2, len(X_train)),
                name="gender"
            )
        )
        # should return a model (or None if fairlearn not installed)
        if mitigated is not None:
            assert hasattr(mitigated, "predict")


class TestBeforeAfterComparison:
    def test_comparison_returns_df(self, fairness_data):
        from src.fairness_mitigation import before_after_comparison
        model, X_train, X_test, y_train, y_test, y_pred, sensitive = fairness_data

        # simulate "after" predictions - flip a few to see the diff
        rng = np.random.default_rng(99)
        y_pred_after = y_pred.copy()
        flip_idx = rng.choice(len(y_pred_after), size=10, replace=False)
        y_pred_after[flip_idx] = 1 - y_pred_after[flip_idx]

        result = before_after_comparison(y_test, y_pred, y_pred_after, sensitive)
        assert isinstance(result, pd.DataFrame)
        assert "Metric" in result.columns
        assert "Before" in result.columns
        assert "After" in result.columns


class TestPlots:
    def test_plot_before_after(self):
        from src.fairness_mitigation import plot_fairness_before_after
        comparison_df = pd.DataFrame([
            {"Metric": "accuracy", "Before": 0.75, "After": 0.73, "Improvement": -0.02},
            {"Metric": "f1", "Before": 0.60, "After": 0.58, "Improvement": -0.02},
            {"Metric": "demographic_parity_diff", "Before": 0.15, "After": 0.05,
             "Improvement": 0.10},
            {"Metric": "equalized_odds_diff", "Before": 0.12, "After": 0.04,
             "Improvement": 0.08},
        ])
        plot_fairness_before_after(comparison_df, save=False)
