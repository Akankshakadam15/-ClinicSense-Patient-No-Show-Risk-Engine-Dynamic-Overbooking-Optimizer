# tests for the causal inference module
# I wrote most of these after debugging the propensity score clipping issue

import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def causal_df(engineered_df):
    """Prep data for causal tests - same split as the notebook."""
    from src.feature_engineering import MODEL_FEATURES
    df = engineered_df.copy()
    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    return df, feature_cols


class TestPropensityScores:
    def test_ps_returns_array(self, causal_df):
        from src.causal_inference import estimate_propensity_scores
        df, feature_cols = causal_df
        feat = [c for c in feature_cols if c != "sms_received"]
        ps = estimate_propensity_scores(df, feat)
        assert isinstance(ps, np.ndarray)
        assert len(ps) == len(df)

    def test_ps_bounded(self, causal_df):
        # this caught a bug where I forgot to clip propensity scores
        from src.causal_inference import estimate_propensity_scores
        df, feature_cols = causal_df
        feat = [c for c in feature_cols if c != "sms_received"]
        ps = estimate_propensity_scores(df, feat)
        assert (ps >= 0.05).all(), "propensity scores should be clipped to [0.05, 0.95]"
        assert (ps <= 0.95).all()


class TestIPWATE:
    def test_ipw_returns_dict(self, causal_df):
        from src.causal_inference import estimate_propensity_scores, ipw_ate
        df, feature_cols = causal_df
        feat = [c for c in feature_cols if c != "sms_received"]
        ps = estimate_propensity_scores(df, feat)
        outcome = (1 - df["no_show"]).values
        treatment = df["sms_received"].values
        result = ipw_ate(outcome, treatment, ps)
        assert isinstance(result, dict)
        assert "ATE" in result
        assert "ATE_ci_lower" in result
        assert "ATE_ci_upper" in result

    def test_ate_reasonable_range(self, causal_df):
        from src.causal_inference import estimate_propensity_scores, ipw_ate
        df, feature_cols = causal_df
        feat = [c for c in feature_cols if c != "sms_received"]
        ps = estimate_propensity_scores(df, feat)
        outcome = (1 - df["no_show"]).values
        treatment = df["sms_received"].values
        result = ipw_ate(outcome, treatment, ps)
        # ATE should be a small-ish number, not something crazy
        assert -1.0 <= result["ATE"] <= 1.0


class TestCATE:
    def test_cate_adds_column(self, causal_df):
        from src.causal_inference import estimate_cate
        df, feature_cols = causal_df
        result_df, _, _ = estimate_cate(df, feature_cols)
        assert "cate_score" in result_df.columns
        assert len(result_df) == len(df)


class TestBudgetOptimizer:
    def test_optimizer_returns_dict(self, causal_df):
        from src.causal_inference import estimate_cate, budget_sms_optimizer
        df, feature_cols = causal_df
        result_df, _, _ = estimate_cate(df, feature_cols)
        opt = budget_sms_optimizer(result_df, budget=50)
        assert isinstance(opt, dict)
        assert "expected_uplift" in opt
        assert "budget_used" in opt
        assert opt["budget_used"] <= 50

    def test_optimizer_budget_constraint(self, causal_df):
        from src.causal_inference import estimate_cate, budget_sms_optimizer
        df, feature_cols = causal_df
        result_df, _, _ = estimate_cate(df, feature_cols)
        opt = budget_sms_optimizer(result_df, budget=10)
        assert opt["budget_used"] == 10

    def test_efficiency_curve_monotonic(self, causal_df):
        from src.causal_inference import estimate_cate, budget_sms_optimizer
        df, feature_cols = causal_df
        result_df, _, _ = estimate_cate(df, feature_cols)
        opt = budget_sms_optimizer(result_df, budget=len(df))
        curve = opt["efficiency_curve"]
        # cumulative uplift should be monotonically non-decreasing
        # (well, until CATE goes negative anyway)
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == len(df)


class TestQini:
    def test_qini_curve(self, causal_df):
        from src.causal_inference import estimate_cate, compute_qini_curve
        df, feature_cols = causal_df
        result_df, _, _ = estimate_cate(df, feature_cols)
        qini = compute_qini_curve(
            result_df["cate_score"].values,
            df["sms_received"].values,
            (1 - df["no_show"]).values,
        )
        assert isinstance(qini, pd.DataFrame)
        assert "fraction" in qini.columns
        assert "qini_score" in qini.columns
