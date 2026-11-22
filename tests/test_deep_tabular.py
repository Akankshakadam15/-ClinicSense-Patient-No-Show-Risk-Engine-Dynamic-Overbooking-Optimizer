"""
Tests for the deep tabular models & statistical comparison module.
McNemar's and DeLong's tests are the important ones here -
they tell us if differences between models are real or just noise.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class TestMcNemarTest:
    def test_returns_dict(self):
        from src.deep_tabular import mcnemar_test
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_pred_a = rng.integers(0, 2, 200)
        y_pred_b = rng.integers(0, 2, 200)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b, "ModelA", "ModelB")
        assert isinstance(result, dict)
        assert "chi2" in result
        assert "p_value" in result

    def test_identical_preds_not_significant(self):
        """if both models predict the same thing, p must be 1.0"""
        from src.deep_tabular import mcnemar_test
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        result = mcnemar_test(y_true, y_pred, y_pred, "A", "B")
        assert result["p_value"] == 1.0
        assert not result["significant_at_005"]

    def test_p_value_bounded(self):
        from src.deep_tabular import mcnemar_test
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_pred_a = rng.integers(0, 2, 200)
        y_pred_b = rng.integers(0, 2, 200)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert 0 <= result["p_value"] <= 1


class TestDeLongTest:
    def test_returns_dict(self):
        from src.deep_tabular import delong_test
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_score_a = rng.uniform(0, 1, 200)
        y_score_b = rng.uniform(0, 1, 200)
        result = delong_test(y_true, y_score_a, y_score_b, "A", "B")
        assert isinstance(result, dict)
        assert "auc_a" in result
        assert "auc_b" in result
        assert "p_value" in result

    def test_same_scores_high_pvalue(self):
        from src.deep_tabular import delong_test
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        scores = rng.uniform(0, 1, 200)
        result = delong_test(y_true, scores, scores)
        # identical scores => no diff => p should be ~1
        assert result["p_value"] >= 0.99


class TestMLPModel:
    def test_mlp_builds_pipeline(self):
        from src.deep_tabular import build_mlp_model
        model = build_mlp_model()
        assert model is not None
        assert hasattr(model, "fit")

    def test_mlp_trains_and_predicts(self, split_data):
        from src.deep_tabular import build_mlp_model
        X_train, X_test, y_train, y_test = split_data
        model = build_mlp_model()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)
        assert (probs >= 0).all() and (probs <= 1).all()

    # def test_tabnet_integration(self, split_data):
    #     """skipping for now - TabNet needs pytorch installed and
    #        it makes the CI pipeline really slow"""
    #     pass


class TestPairwiseTests:
    def test_pairwise_returns_dataframe(self, split_data):
        from src.deep_tabular import pairwise_model_tests
        X_train, X_test, y_train, y_test = split_data

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        preds = {
            "RF": rf.predict(X_test),
            "LR": lr.predict(X_test),
        }
        probs = {
            "RF": rf.predict_proba(X_test)[:, 1],
            "LR": lr.predict_proba(X_test)[:, 1],
        }

        result = pairwise_model_tests(y_test.values, preds, probs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # 2 models -> 1 pair
        assert "DeLong p" in result.columns
        assert "McNemar p" in result.columns
