"""
ClinicSense – Uncertainty-Aware Overbooking Optimizer
Combines my conformal prediction sets (already built in Notebook 07) with
a Newsvendor-style cost model to choose the optimal overbooking level.

Idea: I already have prediction intervals from MAPIE.  The Newsvendor
model (from operations research) answers "how much extra stock to order
when demand is uncertain" – I adapt it to clinic slots.

Ref: Newsvendor formulation from Hillier & Lieberman, Intro to OR Ch. 18
     Conformal prediction: Vovk et al. (2005)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings

from src.config import (
    FIGURES_DIR, RANDOM_STATE,
    COST_IDLE_SLOT, COST_WAITING_MINUTE, COST_OVERTIME_MINUTE,
    SLOT_DURATION_MIN, SLOTS_PER_DAY, SIM_DAYS,
)

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

rng_global = np.random.default_rng(RANDOM_STATE)


# --- conformal attendance bounds ---
def conformal_attendance_bounds(pred_probs: np.ndarray,
                                 conformal_sets: np.ndarray) -> dict:
    """
    From conformal prediction sets, derive bounds on the number of attendees
    for a group of patients.

    Parameters
    ----------
    pred_probs     : array of predicted attendance probabilities (1 - no_show_prob)
    conformal_sets : boolean array (n, 2) – True where class is in prediction set
                     column 0 = class "show", column 1 = class "no-show"

    Returns
    -------
    dict with:
      - 'expected_shows'    : point estimate of total shows
      - 'lower_bound_shows' : conservative lower bound
      - 'upper_bound_shows' : optimistic upper bound
      - 'uncertainty_width' : upper - lower
    """
    n = len(pred_probs)
    expected = float(pred_probs.sum())

    # Lower bound: count patients who are ONLY predicted to show (singleton {show})
    # Upper bound: count patients who MIGHT show (show is in their prediction set)
    if conformal_sets is not None and conformal_sets.shape[1] >= 2:
        certain_show = np.sum(conformal_sets[:, 0] & ~conformal_sets[:, 1])
        might_show = np.sum(conformal_sets[:, 0])
        lower = float(certain_show)
        upper = float(might_show)
    else:
        # Fallback: use probability bounds
        lower = float(np.sum(pred_probs > 0.7))
        upper = float(np.sum(pred_probs > 0.3))

    return {
        "expected_shows": expected,
        "lower_bound_shows": lower,
        "upper_bound_shows": upper,
        "uncertainty_width": upper - lower,
    }


# --- newsvendor overbooking optimiser ---
# CR = c_u / (c_u + c_o)  where c_u = underage cost, c_o = overage cost
def newsvendor_optimal_bookings(expected_shows: float,
                                lower_shows: float,
                                upper_shows: float,
                                n_slots: int = SLOTS_PER_DAY,
                                cost_idle: float = COST_IDLE_SLOT,
                                cost_overflow_per_patient: float = None) -> dict:
    """
    Solve the Newsvendor problem for optimal overbooking.

    The critical ratio determines how aggressively to overbook:
      CR = cost_idle / (cost_idle + cost_overflow)

    If CR > 0.5, idle cost dominates → overbook more.
    If CR < 0.5, overflow cost dominates → overbook less.

    Parameters
    ----------
    expected_shows : expected number of patients who will show up
    lower_shows    : lower bound on shows (from conformal prediction)
    upper_shows    : upper bound on shows
    n_slots        : available slots
    cost_idle      : cost per unused slot
    cost_overflow_per_patient : cost per patient beyond capacity

    Returns
    -------
    dict with optimal bookings and diagnostics
    """
    if cost_overflow_per_patient is None:
        cost_overflow_per_patient = (
            COST_WAITING_MINUTE * SLOT_DURATION_MIN +
            COST_OVERTIME_MINUTE * SLOT_DURATION_MIN
        )

    # Critical ratio
    cr = cost_idle / (cost_idle + cost_overflow_per_patient)

    # Expected no-shows from the interval
    expected_noshows_low = max(0, n_slots - upper_shows)   # optimistic
    expected_noshows_high = max(0, n_slots - lower_shows)  # pessimistic

    # Optimal extra bookings = quantile of the no-show distribution
    # Using the conformal interval as a triangular distribution
    mean_noshows = max(0, n_slots + 1 - expected_shows)

    # Newsvendor solution: book at the CR quantile
    if upper_shows > lower_shows:
        # Linear interpolation within the conformal interval
        optimal_show_target = lower_shows + cr * (upper_shows - lower_shows)
    else:
        optimal_show_target = expected_shows

    optimal_total_bookings = max(n_slots, round(
        n_slots + (n_slots - optimal_show_target)
    ))
    optimal_extra = max(0, optimal_total_bookings - n_slots)

    return {
        "optimal_extra_bookings": int(optimal_extra),
        "optimal_total_bookings": int(optimal_total_bookings),
        "critical_ratio": float(cr),
        "expected_shows": float(expected_shows),
        "conformal_lower": float(lower_shows),
        "conformal_upper": float(upper_shows),
        "cost_idle": float(cost_idle),
        "cost_overflow": float(cost_overflow_per_patient),
    }


# --- day-level simulation with conformal sets ---
def simulate_uncertainty_day(show_probs: np.ndarray,
                             conformal_sets: np.ndarray,
                             n_slots: int = SLOTS_PER_DAY) -> dict:
    """
    Simulate one clinic day using uncertainty-aware overbooking.

    Returns the Newsvendor decision and the realised cost.
    """
    bounds = conformal_attendance_bounds(show_probs, conformal_sets)
    nv = newsvendor_optimal_bookings(
        bounds["expected_shows"],
        bounds["lower_bound_shows"],
        bounds["upper_bound_shows"],
        n_slots=n_slots,
    )

    # Realise the day: sample attendance
    rng = np.random.default_rng()
    n_book = min(len(show_probs), nv["optimal_total_bookings"])
    chosen = rng.choice(len(show_probs), size=n_book, replace=False)
    attend = rng.random(n_book) < show_probs[chosen]
    n_showed = int(attend.sum())

    idle = max(0, n_slots - n_showed)
    overflow = max(0, n_showed - n_slots)

    idle_cost = idle * COST_IDLE_SLOT
    wait_cost = overflow * SLOT_DURATION_MIN * COST_WAITING_MINUTE
    ot_cost = overflow * SLOT_DURATION_MIN * COST_OVERTIME_MINUTE

    return {
        "extra_booked": nv["optimal_extra_bookings"],
        "n_booked": n_book,
        "n_showed": n_showed,
        "idle_slots": idle,
        "overflow": overflow,
        "total_cost": float(idle_cost + wait_cost + ot_cost),
        "utilisation": float(min(n_showed, n_slots) / n_slots),
        **nv,
    }


# --- MC comparison: does conformal-aware overbooking actually help? ---
def compare_with_uncertainty(daily_probs: list,
                             daily_conformal_sets: list = None,
                             n_slots: int = SLOTS_PER_DAY,
                             n_days: int = 1000) -> pd.DataFrame:
    """
    Run Monte-Carlo simulation comparing:
      - No overbooking
      - Fixed +2
      - Risk-based dynamic
      - Newsvendor (uncertainty-aware)

    Parameters
    ----------
    daily_probs          : list of arrays of attendance probabilities
    daily_conformal_sets : list of conformal prediction set arrays (or None)
    n_slots              : base slots per day
    n_days               : simulation replications

    Returns
    -------
    summary DataFrame indexed by strategy name
    """
    rng = np.random.default_rng(RANDOM_STATE)

    strategies = {
        "No Overbooking": lambda probs, cs: 0,
        "Fixed +2": lambda probs, cs: 2,
        "Risk-Based Dynamic": lambda probs, cs: max(0, round(np.sum(1 - probs))),
    }

    def newsvendor_fn(probs, cs):
        bounds = conformal_attendance_bounds(probs, cs)
        nv = newsvendor_optimal_bookings(
            bounds["expected_shows"],
            bounds["lower_bound_shows"],
            bounds["upper_bound_shows"],
            n_slots=n_slots,
        )
        return nv["optimal_extra_bookings"]

    strategies["Newsvendor (Conformal)"] = newsvendor_fn

    results = {name: [] for name in strategies}

    for day_i in range(n_days):
        pool_idx = day_i % len(daily_probs)
        probs = daily_probs[pool_idx]
        cs = daily_conformal_sets[pool_idx] if daily_conformal_sets else None

        for name, fn in strategies.items():
            extra = fn(probs, cs)
            n_book = min(len(probs), n_slots + extra)
            chosen = rng.choice(len(probs), size=n_book, replace=False)
            attend = rng.random(n_book) < probs[chosen]
            n_showed = int(attend.sum())

            idle = max(0, n_slots - n_showed)
            overflow = max(0, n_showed - n_slots)
            cost = (idle * COST_IDLE_SLOT +
                    overflow * SLOT_DURATION_MIN * COST_WAITING_MINUTE +
                    overflow * SLOT_DURATION_MIN * COST_OVERTIME_MINUTE)

            results[name].append({
                "total_cost": cost,
                "idle_slots": idle,
                "overflow": overflow,
                "utilisation": min(n_showed, n_slots) / n_slots,
            })

    summary_rows = []
    for name, recs in results.items():
        df = pd.DataFrame(recs)
        summary_rows.append({
            "Strategy": name,
            "Avg Total Cost": df["total_cost"].mean(),
            "Std Cost": df["total_cost"].std(),
            "Avg Idle Slots": df["idle_slots"].mean(),
            "Avg Overflow": df["overflow"].mean(),
            "Avg Utilisation %": df["utilisation"].mean() * 100,
        })

    return pd.DataFrame(summary_rows).set_index("Strategy").round(2)


# --- sensitivity: what happens as we vary the confidence level? ---
def sensitivity_over_confidence(model, X_cal, y_cal, X_test,
                                 confidence_levels: list = None) -> pd.DataFrame:
    """
    Analyse how different conformal confidence levels affect the
    Newsvendor decision.

    Returns DataFrame with columns: confidence, avg_extra, avg_width
    """
    if confidence_levels is None:
        confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]

    from src.advanced_analytics import conformal_predict

    records = []
    for conf in confidence_levels:
        pred_sets, coverage, singleton_rate = conformal_predict(
            model, X_cal, y_cal, X_test, confidence=conf
        )
        if pred_sets is not None:
            probs = model.predict_proba(X_test)[:, 0]  # P(show)
            bounds = conformal_attendance_bounds(probs, pred_sets)
            nv = newsvendor_optimal_bookings(
                bounds["expected_shows"],
                bounds["lower_bound_shows"],
                bounds["upper_bound_shows"],
            )
            records.append({
                "confidence": conf,
                "optimal_extra": nv["optimal_extra_bookings"],
                "uncertainty_width": bounds["uncertainty_width"],
                "coverage": coverage,
                "singleton_rate": singleton_rate,
            })

    return pd.DataFrame(records)


# --- plots ---
def plot_uncertainty_comparison(summary_df: pd.DataFrame, save: bool = True):
    """Plot strategy comparison including the Newsvendor approach."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    summary_df["Avg Total Cost"].plot(kind="bar", ax=axes[0],
                                       color="steelblue", edgecolor="black")
    axes[0].set_title("Avg Total Cost per Day", fontsize=12)
    axes[0].set_ylabel("Cost (₹)")
    axes[0].tick_params(axis="x", rotation=45)

    summary_df["Avg Idle Slots"].plot(kind="bar", ax=axes[1],
                                       color="tomato", edgecolor="black")
    axes[1].set_title("Avg Idle Slots per Day", fontsize=12)
    axes[1].set_ylabel("Slots")
    axes[1].tick_params(axis="x", rotation=45)

    summary_df["Avg Utilisation %"].plot(kind="bar", ax=axes[2],
                                          color="mediumseagreen", edgecolor="black")
    axes[2].set_title("Avg Utilisation (%)", fontsize=12)
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle("Uncertainty-Aware Overbooking Comparison", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "uncertainty_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence_sensitivity(sensitivity_df: pd.DataFrame, save: bool = True):
    """Plot how confidence level affects the Newsvendor decision."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sensitivity_df["confidence"], sensitivity_df["optimal_extra"],
                 "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_xlabel("Confidence Level", fontsize=12)
    axes[0].set_ylabel("Optimal Extra Bookings", fontsize=12)
    axes[0].set_title("Newsvendor Decision vs Confidence", fontsize=13)
    axes[0].grid(alpha=0.3)

    axes[1].plot(sensitivity_df["confidence"], sensitivity_df["uncertainty_width"],
                 "s-", color="darkorange", linewidth=2, markersize=8)
    axes[1].set_xlabel("Confidence Level", fontsize=12)
    axes[1].set_ylabel("Prediction Set Width", fontsize=12)
    axes[1].set_title("Conformal Uncertainty vs Confidence", fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Sensitivity Analysis – Conformal Confidence Level",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "confidence_sensitivity.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
