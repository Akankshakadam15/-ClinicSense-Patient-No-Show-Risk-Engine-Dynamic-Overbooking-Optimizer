"""
ClinicSense – Survival Analysis
I reframed lead_time_days as a time-to-event variable so I can apply
Kaplan-Meier and Cox PH models.  The key question is: "beyond how many
days of lead time does the no-show risk start rising sharply?"

Ref: Kleinbaum & Klein, Survival Analysis (3rd ed.), Ch. 1-2
     lifelines docs: https://lifelines.readthedocs.io/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings

from src.config import FIGURES_DIR, RANDOM_STATE

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Kaplan-Meier Estimator
# ---------------------------------------------------------------------------
def kaplan_meier_estimate(durations: np.ndarray,
                          events: np.ndarray) -> pd.DataFrame:
    """
    Compute the Kaplan-Meier survival function from raw data.

    Parameters
    ----------
    durations : array of time values (lead_time_days)
    events    : array of event indicators (1 = no-show event, 0 = censored/showed)

    Returns
    -------
    DataFrame with columns: time, n_risk, n_events, survival_prob
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    # get sorted unique event times
    unique_times = np.sort(np.unique(durations[events == 1]))
    if len(unique_times) == 0:
        unique_times = np.sort(np.unique(durations))

    records = []
    # n_total = len(durations)  # not actually used, kept for reference
    survival = 1.0

    for t in unique_times:
        n_risk = np.sum(durations >= t)
        n_events = np.sum((durations == t) & (events == 1))
        if n_risk > 0:
            survival *= (1 - n_events / n_risk)
        records.append({
            "time": t,
            "n_risk": int(n_risk),
            "n_events": int(n_events),
            "survival_prob": survival,
        })

    return pd.DataFrame(records)


def plot_kaplan_meier(df: pd.DataFrame,
                     duration_col: str = "lead_time_days",
                     event_col: str = "no_show",
                     group_col: str = None,
                     title: str = "Kaplan-Meier Survival Curve",
                     save: bool = True) -> pd.DataFrame:
    """
    Plot Kaplan-Meier survival curves, optionally stratified by group_col.

    Returns the KM table(s) as a DataFrame.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if group_col is None:
        km = kaplan_meier_estimate(df[duration_col].values, df[event_col].values)
        ax.step(km["time"], km["survival_prob"], where="post",
                linewidth=2, color="steelblue", label="All Patients")
        result = km
    else:
        groups = sorted(df[group_col].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(groups), 2)))
        results = []
        for i, g in enumerate(groups):
            mask = df[group_col] == g
            km = kaplan_meier_estimate(
                df.loc[mask, duration_col].values,
                df.loc[mask, event_col].values,
            )
            km["group"] = g
            results.append(km)
            ax.step(km["time"], km["survival_prob"], where="post",
                    linewidth=2, color=colors[i], label=f"{group_col}={g}")
        result = pd.concat(results, ignore_index=True)

    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("Survival Probability (Attendance)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)  # TODO: add 95% CI band like in the textbook example
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_").replace("–", "").replace("-", "_")
        path = os.path.join(FIGURES_DIR, f"km_{fname}.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 2. Cox Proportional Hazards  (semi-parametric)
# ---------------------------------------------------------------------------
def cox_proportional_hazards(df: pd.DataFrame,
                              duration_col: str = "lead_time_days",
                              event_col: str = "no_show",
                              feature_cols: list = None) -> dict:
    """
    Fit a Cox PH model using lifelines.

    Returns
    -------
    dict with keys:
      - 'model'   : fitted CoxPHFitter
      - 'summary' : DataFrame of coefficients, HR, p-values
      - 'concordance' : concordance index (C-index)
    """
    try:
        from lifelines import CoxPHFitter

        if feature_cols is None:
            exclude = {duration_col, event_col, "patient_id", "appointment_id",
                       "scheduled_day", "appointment_day", "age_group",
                       "cluster", "cluster_name", "uplift_score", "attended"}
            feature_cols = [c for c in df.columns
                           if c not in exclude and df[c].dtype != object]

        cols = feature_cols + [duration_col, event_col]
        cox_df = df[cols].dropna().copy()

        # Cox blows up if duration=0, so clip to half a day
        # (I spent an hour debugging this before I realised...)
        cox_df[duration_col] = cox_df[duration_col].clip(lower=0.5)

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cox_df, duration_col=duration_col, event_col=event_col)

        summary = cph.summary.copy()
        summary["hazard_ratio"] = np.exp(summary["coef"])

        return {
            "model": cph,
            "summary": summary,
            "concordance": cph.concordance_index_,
        }

    except ImportError:
        print("[Survival] lifelines not installed. Run: pip install lifelines")
        return {"model": None, "summary": None, "concordance": None}


def plot_cox_coefficients(cox_result: dict, save: bool = True):
    """Forest plot of Cox PH hazard ratios."""
    summary = cox_result.get("summary")
    if summary is None:
        print("[Survival] No Cox summary to plot.")
        return

    hr = summary["hazard_ratio"].sort_values()
    ci_lower = np.exp(summary.loc[hr.index, "coef lower 95%"])
    ci_upper = np.exp(summary.loc[hr.index, "coef upper 95%"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(hr) * 0.4)))
    y_pos = range(len(hr))

    ax.barh(y_pos, hr.values - 1, left=1, height=0.5,
            color=["#F44336" if h > 1 else "#4CAF50" for h in hr.values],
            alpha=0.7, edgecolor="black")

    # Error bars
    for i, (feat, h) in enumerate(hr.items()):
        ax.plot([ci_lower[feat], ci_upper[feat]], [i, i],
                color="black", linewidth=1.5)

    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(hr.index, fontsize=10)
    ax.set_xlabel("Hazard Ratio (exp(β))", fontsize=12)
    ax.set_title(f"Cox PH – Hazard Ratios  (C-index={cox_result['concordance']:.3f})",
                 fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "cox_hazard_ratios.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Lead-time risk threshold
#    "At which day does the no-show rate cross 25 %?"  <-- this is the key
# ---------------------------------------------------------------------------
def find_lead_time_risk_threshold(df: pd.DataFrame,
                                   duration_col: str = "lead_time_days",
                                   event_col: str = "no_show",
                                   granularity: int = 1) -> dict:
    """
    Find the lead-time day where the no-show rate crosses key thresholds.

    Returns
    -------
    dict with:
      - 'daily_rates'     : DataFrame of day-by-day no-show rates
      - 'risk_onset_day'  : first day where rate > 25%
      - 'high_risk_day'   : first day where rate > 40%
      - 'median_rate'     : overall median daily no-show rate
    """
    grouped = (
        df.groupby(duration_col)[event_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "no_show_rate", "count": "n_appointments"})
        .reset_index()
    )
    grouped = grouped[grouped["n_appointments"] >= 5]  # filter noise

    risk_onset = grouped.loc[grouped["no_show_rate"] > 0.25, duration_col]
    high_risk  = grouped.loc[grouped["no_show_rate"] > 0.40, duration_col]

    return {
        "daily_rates": grouped,
        "risk_onset_day": int(risk_onset.min()) if len(risk_onset) > 0 else None,
        "high_risk_day": int(high_risk.min()) if len(high_risk) > 0 else None,
        "median_rate": float(grouped["no_show_rate"].median()),
    }


def plot_lead_time_hazard(df: pd.DataFrame,
                          duration_col: str = "lead_time_days",
                          event_col: str = "no_show",
                          save: bool = True):
    """Plot the no-show rate as a function of lead time."""
    result = find_lead_time_risk_threshold(df, duration_col, event_col)
    rates = result["daily_rates"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(rates[duration_col], rates["no_show_rate"],
           color="steelblue", alpha=0.7, edgecolor="white", label="No-Show Rate")

    # Overlay smoothed trend
    if len(rates) > 3:
        rolling = rates.set_index(duration_col)["no_show_rate"].rolling(
            window=5, min_periods=1, center=True
        ).mean()
        ax.plot(rolling.index, rolling.values, color="red", linewidth=2.5,
                label="5-day Rolling Avg")

    # Mark thresholds
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=1.5,
               label="25% Risk Threshold")
    ax.axhline(0.40, color="red", linestyle="--", linewidth=1.5,
               label="40% High-Risk Threshold")

    if result["risk_onset_day"] is not None:
        ax.axvline(result["risk_onset_day"], color="orange", linestyle=":",
                   alpha=0.7)

    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("No-Show Rate", fontsize=12)
    ax.set_title("No-Show Hazard by Lead Time", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-1, min(90, rates[duration_col].max() + 1))
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "lead_time_hazard.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 4. Nelson-Aalen cumulative hazard  (non-parametric alternative)
# ---------------------------------------------------------------------------
def nelson_aalen_estimate(durations: np.ndarray,
                          events: np.ndarray) -> pd.DataFrame:
    """
    Compute the Nelson-Aalen cumulative hazard estimator.

    Returns DataFrame with columns: time, cumulative_hazard
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    unique_times = np.sort(np.unique(durations[events == 1]))
    if len(unique_times) == 0:
        unique_times = np.sort(np.unique(durations))

    records = []
    cum_hazard = 0.0

    for t in unique_times:
        n_risk = np.sum(durations >= t)
        n_events = np.sum((durations == t) & (events == 1))
        if n_risk > 0:
            cum_hazard += n_events / n_risk
        records.append({
            "time": t,
            "cumulative_hazard": cum_hazard,
        })

    return pd.DataFrame(records)


def plot_nelson_aalen(df: pd.DataFrame,
                      duration_col: str = "lead_time_days",
                      event_col: str = "no_show",
                      save: bool = True):
    """Plot the Nelson-Aalen cumulative hazard."""
    na = nelson_aalen_estimate(df[duration_col].values, df[event_col].values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(na["time"], na["cumulative_hazard"], where="post",
            linewidth=2, color="darkred")
    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("Cumulative Hazard", fontsize=12)
    ax.set_title("Nelson-Aalen Cumulative Hazard Estimate", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "nelson_aalen_hazard.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return na
