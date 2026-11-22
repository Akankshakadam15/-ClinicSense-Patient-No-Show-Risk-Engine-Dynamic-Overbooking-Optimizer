"""
Quick tests for the survival analysis functions.
Mostly checking output shapes, dtypes and basic monotonicity.
Ref: Kleinbaum & Klein Ch.1 for why survival curves must be non-increasing.
"""

import pytest
import pandas as pd
import numpy as np


class TestKaplanMeier:
    def test_km_returns_dataframe(self, engineered_df):
        from src.survival import kaplan_meier_estimate
        km = kaplan_meier_estimate(
            engineered_df["lead_time_days"].values,
            engineered_df["no_show"].values,
        )
        assert isinstance(km, pd.DataFrame)
        assert "time" in km.columns
        assert "survival_prob" in km.columns

    def test_km_survival_decreasing(self, engineered_df):
        """survival curve must be non-increasing - caught a sorting bug here once"""
        from src.survival import kaplan_meier_estimate
        km = kaplan_meier_estimate(
            engineered_df["lead_time_days"].values,
            engineered_df["no_show"].values,
        )
        probs = km["survival_prob"].values
        assert all(probs[i] >= probs[i+1] - 1e-9 for i in range(len(probs)-1))

    def test_km_survival_between_0_and_1(self, engineered_df):
        from src.survival import kaplan_meier_estimate
        km = kaplan_meier_estimate(
            engineered_df["lead_time_days"].values,
            engineered_df["no_show"].values,
        )
        # sanity check - probabilities must be valid
        assert (km["survival_prob"] >= 0).all()
        assert (km["survival_prob"] <= 1).all()

    def test_plot_km_runs(self, engineered_df):
        from src.survival import plot_kaplan_meier
        result = plot_kaplan_meier(engineered_df, save=False)
        assert isinstance(result, pd.DataFrame)

    def test_plot_km_stratified(self, engineered_df):
        from src.survival import plot_kaplan_meier
        result = plot_kaplan_meier(
            engineered_df, group_col="sms_received", save=False
        )
        assert "group" in result.columns

    # def test_cox_with_missing_values(self, engineered_df):
    #     """leaving this for later - need to handle NaNs in lead_time first"""
    #     pass


class TestNelsonAalen:
    def test_na_returns_dataframe(self, engineered_df):
        from src.survival import nelson_aalen_estimate
        na = nelson_aalen_estimate(
            engineered_df["lead_time_days"].values,
            engineered_df["no_show"].values,
        )
        assert isinstance(na, pd.DataFrame)
        assert "cumulative_hazard" in na.columns

    def test_na_cumhazard_increasing(self, engineered_df):
        from src.survival import nelson_aalen_estimate
        na = nelson_aalen_estimate(
            engineered_df["lead_time_days"].values,
            engineered_df["no_show"].values,
        )
        hazard = na["cumulative_hazard"].values
        # cumulative hazard should only go up
        assert all(hazard[i] <= hazard[i+1] + 1e-9 for i in range(len(hazard)-1))


class TestLeadTimeThreshold:
    def test_returns_dict(self, engineered_df):
        from src.survival import find_lead_time_risk_threshold
        result = find_lead_time_risk_threshold(engineered_df)
        assert isinstance(result, dict)
        assert "daily_rates" in result
        assert "median_rate" in result

    def test_daily_rates_is_dataframe(self, engineered_df):
        from src.survival import find_lead_time_risk_threshold
        result = find_lead_time_risk_threshold(engineered_df)
        assert isinstance(result["daily_rates"], pd.DataFrame)

    def test_plot_hazard_runs(self, engineered_df):
        from src.survival import plot_lead_time_hazard
        result = plot_lead_time_hazard(engineered_df, save=False)
        assert isinstance(result, dict)
    # TODO: add a test for edge case where all patients show up
