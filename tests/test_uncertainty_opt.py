"""
Tests for the uncertainty-aware optimization module.
The newsvendor math was tricky to get right so these tests
check the edge cases carefully.
"""

import pytest
import pandas as pd
import numpy as np


class TestConformalBounds:
    def test_bounds_returns_dict(self):
        from src.uncertainty_optimization import conformal_attendance_bounds
        probs = np.array([0.8, 0.9, 0.7, 0.6, 0.95])
        # Simulate conformal sets: (n, 2) boolean array
        sets = np.array([
            [True, False],   # only show
            [True, False],   # only show
            [True, True],    # uncertain
            [True, True],    # uncertain
            [True, False],   # only show
        ])
        result = conformal_attendance_bounds(probs, sets)
        assert isinstance(result, dict)
        assert "expected_shows" in result
        assert "lower_bound_shows" in result
        assert "upper_bound_shows" in result

    def test_bounds_lower_le_upper(self):
        from src.uncertainty_optimization import conformal_attendance_bounds
        probs = np.array([0.8, 0.3, 0.9, 0.5])
        sets = np.array([
            [True, False],
            [True, True],
            [True, False],
            [False, True],
        ])
        result = conformal_attendance_bounds(probs, sets)
        assert result["lower_bound_shows"] <= result["upper_bound_shows"], \
            "lower bound can't exceed upper bound!"

    def test_bounds_with_none_conformal(self):
        # when conformal sets aren't available, should still work
        from src.uncertainty_optimization import conformal_attendance_bounds
        probs = np.array([0.8, 0.3, 0.9, 0.5])
        result = conformal_attendance_bounds(probs, None)
        assert isinstance(result, dict)


class TestNewsvendor:
    def test_newsvendor_returns_dict(self):
        from src.uncertainty_optimization import newsvendor_optimal_bookings
        result = newsvendor_optimal_bookings(
            expected_shows=16.0,
            lower_shows=14.0,
            upper_shows=18.0,
            n_slots=20,
        )
        assert isinstance(result, dict)
        assert "optimal_extra_bookings" in result
        assert "critical_ratio" in result

    def test_newsvendor_extra_nonneg(self):
        from src.uncertainty_optimization import newsvendor_optimal_bookings
        result = newsvendor_optimal_bookings(
            expected_shows=22.0,
            lower_shows=20.0,
            upper_shows=24.0,
            n_slots=20,
        )
        # can't have negative extra bookings
        assert result["optimal_extra_bookings"] >= 0

    def test_critical_ratio_between_0_and_1(self):
        from src.uncertainty_optimization import newsvendor_optimal_bookings
        result = newsvendor_optimal_bookings(
            expected_shows=16.0, lower_shows=14.0, upper_shows=18.0)
        assert 0 <= result["critical_ratio"] <= 1


class TestComparison:
    def test_compare_returns_dataframe(self):
        from src.uncertainty_optimization import compare_with_uncertainty
        rng = np.random.default_rng(42)
        pools = [rng.uniform(0.6, 0.95, 30) for _ in range(50)]
        result = compare_with_uncertainty(pools, n_days=100)
        assert isinstance(result, pd.DataFrame)
        assert "Avg Total Cost" in result.columns
        assert len(result) == 4  # 4 strategies

    def test_newsvendor_is_competitive(self):
        """newsvendor should be at least in the right ballpark"""
        from src.uncertainty_optimization import compare_with_uncertainty
        rng = np.random.default_rng(42)
        pools = [rng.uniform(0.6, 0.95, 30) for _ in range(50)]
        result = compare_with_uncertainty(pools, n_days=200)
        nv_cost = result.loc["Newsvendor (Conformal)", "Avg Total Cost"]
        no_ob_cost = result.loc["No Overbooking", "Avg Total Cost"]
        assert nv_cost <= no_ob_cost * 1.5
