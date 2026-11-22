"""
ClinicSense – Causal Inference & SMS Budget Optimizer
This is the prescriptive part of the project.  Instead of just predicting
who will no-show, we ask: "which patients benefit most from an SMS reminder,
and how should we allocate a limited budget of SMS messages?"

Methodology follows:
  - Imbens & Rubin (2015) for IPW / propensity scores
  - Kennedy (2020) for doubly-robust estimation
  - Gutierrez & Gerardy (2017) for Qini curves

I already had the T-Learner in advanced_analytics.py; this extends it
with proper causal identification and a budget-constrained optimizer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from src.config import FIGURES_DIR, RANDOM_STATE

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# --- Propensity scores ---------------------------------------------------------
def estimate_propensity_scores(df: pd.DataFrame,
                                feature_cols: list,
                                treatment_col: str = "sms_received") -> np.ndarray:
    """
    Estimate P(treatment=1 | X) using logistic regression with cross-validation.

    Returns
    -------
    propensity_scores : array of shape (n_samples,)
    """
    X = df[feature_cols].fillna(0)
    T = df[treatment_col].values

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    # Use cross-validated predictions to avoid overfitting
    ps = cross_val_predict(lr, X, T, cv=5, method="predict_proba")[:, 1]

    # clip to [0.05, 0.95] to avoid division-by-near-zero weights
    # (Crump et al. 2009 recommend this trimming range)
    ps = np.clip(ps, 0.05, 0.95)
    return ps


def plot_propensity_overlap(propensity_scores: np.ndarray,
                            treatment: np.ndarray,
                            save: bool = True):
    """Plot propensity score distributions for treated and control groups."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(propensity_scores[treatment == 0], bins=40, alpha=0.6,
            color="steelblue", label="Control (no SMS)", density=True)
    ax.hist(propensity_scores[treatment == 1], bins=40, alpha=0.6,
            color="darkorange", label="Treated (SMS)", density=True)
    ax.set_xlabel("Propensity Score P(SMS=1|X)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Propensity Score Overlap", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "propensity_overlap.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# --- IPW-ATE (Horvitz-Thompson style) ----------------------------------------
def ipw_ate(outcome: np.ndarray,
            treatment: np.ndarray,
            propensity: np.ndarray) -> dict:
    """
    Compute the Average Treatment Effect using Inverse Propensity Weighting.

    outcome   : binary (1 = attended, 0 = no-show)
    treatment : binary (1 = SMS sent)
    propensity: P(treatment=1 | X)

    Returns dict with ATE, ATT, ATC and standard errors.
    """
    T = treatment.astype(float)
    Y = outcome.astype(float)
    e = propensity

    # IPW estimator for ATE
    w1 = T / e
    w0 = (1 - T) / (1 - e)

    ate_est = np.mean(w1 * Y) - np.mean(w0 * Y)

    # bootstrap SE – 500 resamples seems enough, tried 1000 and results barely changed
    n = len(Y)
    rng = np.random.default_rng(RANDOM_STATE)
    boot_ates = []
    for _ in range(500):
        idx = rng.choice(n, size=n, replace=True)
        w1_b = T[idx] / e[idx]
        w0_b = (1 - T[idx]) / (1 - e[idx])
        boot_ates.append(np.mean(w1_b * Y[idx]) - np.mean(w0_b * Y[idx]))
    se = np.std(boot_ates)

    # ATT: Average Treatment Effect on the Treated
    att = np.mean(Y[T == 1]) - np.mean(w0[T == 0] * Y[T == 0]) / np.mean(w0[T == 0])

    return {
        "ATE": float(ate_est),
        "ATE_se": float(se),
        "ATE_ci_lower": float(ate_est - 1.96 * se),
        "ATE_ci_upper": float(ate_est + 1.96 * se),
        "ATT": float(att),
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
    }


# --- Doubly robust ATE --------------------------------------------------------
# consistent if EITHER the propensity or outcome model is correct
def doubly_robust_ate(df: pd.DataFrame,
                      feature_cols: list,
                      treatment_col: str = "sms_received",
                      outcome_col: str = "no_show") -> dict:
    """
    Doubly robust estimator: combines outcome modelling with IPW.
    Consistent if either the propensity or outcome model is correct.

    Returns dict with DR-ATE and bootstrap CI.
    """
    X = df[feature_cols].fillna(0).values
    T = df[treatment_col].values.astype(float)
    # Outcome = attendance (1 - no_show)
    Y = (1 - df[outcome_col].values).astype(float)

    # Propensity model
    ps_model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    e = cross_val_predict(ps_model, X, T.astype(int), cv=5, method="predict_proba")[:, 1]
    e = np.clip(e, 0.05, 0.95)

    # Outcome models
    rf1 = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                      random_state=RANDOM_STATE)
    rf0 = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                      random_state=RANDOM_STATE)

    treated_mask = T == 1
    rf1.fit(X[treated_mask], Y[treated_mask].astype(int))
    rf0.fit(X[~treated_mask], Y[~treated_mask].astype(int))

    mu1 = rf1.predict_proba(X)[:, 1]
    mu0 = rf0.predict_proba(X)[:, 1]

    # DR estimator
    dr_scores = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    dr_ate = np.mean(dr_scores)
    dr_se = np.std(dr_scores) / np.sqrt(len(dr_scores))

    return {
        "DR_ATE": float(dr_ate),
        "DR_ATE_se": float(dr_se),
        "DR_ATE_ci_lower": float(dr_ate - 1.96 * dr_se),
        "DR_ATE_ci_upper": float(dr_ate + 1.96 * dr_se),
    }


# --- CATE (Conditional ATE per patient) --------------------------------------
def estimate_cate(df: pd.DataFrame,
                  feature_cols: list,
                  treatment_col: str = "sms_received",
                  outcome_col: str = "no_show") -> pd.DataFrame:
    """
    Estimate Conditional Average Treatment Effect for each patient
    using a T-Learner with GradientBoosting.

    CATE_i = E[Y(1)|X_i] - E[Y(0)|X_i]
    where Y = attendance (1 - no_show), so positive CATE = SMS helps.

    Returns df with 'cate_score' column added.
    """
    df = df.copy()
    df["attended"] = 1 - df[outcome_col]

    X_cols = [c for c in feature_cols if c != treatment_col]
    X = df[X_cols].fillna(0)

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    clf_t1 = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    clf_t0 = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE
    )

    clf_t1.fit(treated[X_cols].fillna(0), treated["attended"])
    clf_t0.fit(control[X_cols].fillna(0), control["attended"])

    mu1 = clf_t1.predict_proba(X)[:, 1]
    mu0 = clf_t0.predict_proba(X)[:, 1]

    df["cate_score"] = mu1 - mu0
    return df, clf_t1, clf_t0


# --- Budget-constrained optimizer --------------------------------------------
# greedy: just rank patients by CATE and pick top-K
def budget_sms_optimizer(df: pd.DataFrame,
                         budget: int,
                         cate_col: str = "cate_score",
                         cost_per_sms: float = 1.0) -> dict:
    """
    Given a budget of K SMS messages, select the top-K patients by CATE
    to maximise total attendance uplift.

    Parameters
    ----------
    df        : DataFrame with cate_score column
    budget    : maximum number of SMS to send
    cate_col  : column name for CATE scores
    cost_per_sms : cost per SMS (default 1.0 unit)

    Returns
    -------
    dict with:
      - 'selected_indices'    : indices of selected patients
      - 'expected_uplift'     : total expected attendance uplift
      - 'budget_used'         : number of SMS sent
      - 'avg_cate_selected'   : average CATE of selected patients
      - 'avg_cate_unselected' : average CATE of remaining patients
      - 'efficiency_curve'    : DataFrame with cumulative uplift at each budget level
    """
    sorted_df = df.sort_values(cate_col, ascending=False).copy()
    n = len(sorted_df)
    effective_budget = min(budget, n)

    selected = sorted_df.head(effective_budget)
    not_selected = sorted_df.tail(n - effective_budget)

    # Build efficiency curve
    cumulative_uplift = sorted_df[cate_col].cumsum().values
    efficiency = pd.DataFrame({
        "n_sms": range(1, n + 1),
        "cumulative_uplift": cumulative_uplift,
        "marginal_cate": sorted_df[cate_col].values,
    })

    return {
        "selected_indices": selected.index.tolist(),
        "expected_uplift": float(selected[cate_col].sum()),
        "budget_used": effective_budget,
        "total_cost": float(effective_budget * cost_per_sms),
        "avg_cate_selected": float(selected[cate_col].mean()),
        "avg_cate_unselected": float(not_selected[cate_col].mean()) if len(not_selected) > 0 else 0.0,
        "efficiency_curve": efficiency,
    }


def plot_budget_efficiency(optimizer_result: dict,
                           budget_markers: list = None,
                           save: bool = True):
    """Plot the cumulative uplift curve as a function of SMS budget."""
    eff = optimizer_result["efficiency_curve"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative uplift
    axes[0].plot(eff["n_sms"], eff["cumulative_uplift"],
                 color="steelblue", linewidth=2)
    axes[0].set_xlabel("Number of SMS Sent", fontsize=12)
    axes[0].set_ylabel("Cumulative Expected Uplift", fontsize=12)
    axes[0].set_title("SMS Budget vs Attendance Uplift", fontsize=13)
    axes[0].grid(alpha=0.3)

    if budget_markers:
        for b in budget_markers:
            if b < len(eff):
                val = eff.loc[eff["n_sms"] == b, "cumulative_uplift"].values
                if len(val) > 0:
                    axes[0].axvline(b, color="red", linestyle="--", alpha=0.6)
                    axes[0].annotate(f"B={b}: {val[0]:.1f}",
                                     xy=(b, val[0]), fontsize=9)

    # Marginal CATE
    axes[1].bar(eff["n_sms"], eff["marginal_cate"],
                color="darkorange", alpha=0.7, width=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Patient Rank (by CATE)", fontsize=12)
    axes[1].set_ylabel("Individual CATE", fontsize=12)
    axes[1].set_title("Marginal Treatment Effect per Patient", fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Causal SMS Targeting – Budget Efficiency", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "sms_budget_efficiency.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Qini curve (uplift evaluation) ------------------------------------------
def compute_qini_curve(cate_scores: np.ndarray,
                       treatment: np.ndarray,
                       outcome: np.ndarray) -> pd.DataFrame:
    """
    Compute the Qini curve for evaluating uplift model quality.

    Parameters
    ----------
    cate_scores : predicted CATE values
    treatment   : binary treatment indicator
    outcome     : binary outcome (1 = attended)

    Returns DataFrame with columns: fraction, qini_score
    """
    n = len(cate_scores)
    order = np.argsort(-cate_scores)  # descending

    t_sorted = treatment[order]
    y_sorted = outcome[order]

    n_t = treatment.sum()
    n_c = (1 - treatment).sum()

    qini_points = [{"fraction": 0.0, "qini_score": 0.0}]

    for k in range(1, n + 1):
        n_t_k = t_sorted[:k].sum()
        n_c_k = k - n_t_k
        y_t_k = (y_sorted[:k] * t_sorted[:k]).sum()
        y_c_k = (y_sorted[:k] * (1 - t_sorted[:k])).sum()

        if n_c_k > 0 and n_t > 0:
            qini = y_t_k - y_c_k * (n_t_k / max(n_c_k, 1))
        else:
            qini = y_t_k

        if k % max(1, n // 100) == 0 or k == n:
            qini_points.append({
                "fraction": k / n,
                "qini_score": float(qini),
            })

    return pd.DataFrame(qini_points)


def plot_qini_curve(qini_df: pd.DataFrame, save: bool = True):
    """Plot the Qini curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qini_df["fraction"], qini_df["qini_score"],
            color="steelblue", linewidth=2, label="CATE Model")
    ax.plot([0, 1], [0, qini_df["qini_score"].iloc[-1]],
            color="grey", linestyle="--", linewidth=1, label="Random")
    ax.fill_between(qini_df["fraction"], qini_df["qini_score"],
                     np.linspace(0, qini_df["qini_score"].iloc[-1], len(qini_df)),
                     alpha=0.15, color="steelblue")
    ax.set_xlabel("Fraction of Population Targeted", fontsize=12)
    ax.set_ylabel("Qini Score (Incremental Gains)", fontsize=12)
    ax.set_title("Qini Curve – SMS Uplift Model", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "qini_curve.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# --- ATE comparison plot ------------------------------------------------------
def plot_ate_comparison(ipw_result: dict, dr_result: dict, save: bool = True):
    """Bar chart comparing IPW-ATE and DR-ATE with confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["IPW-ATE", "Doubly Robust ATE"]
    estimates = [ipw_result["ATE"], dr_result["DR_ATE"]]
    errors = [
        [ipw_result["ATE"] - ipw_result["ATE_ci_lower"],
         ipw_result["ATE_ci_upper"] - ipw_result["ATE"]],
        [dr_result["DR_ATE"] - dr_result["DR_ATE_ci_lower"],
         dr_result["DR_ATE_ci_upper"] - dr_result["DR_ATE"]],
    ]
    errors_T = list(zip(*errors))

    colors = ["steelblue", "darkorange"]
    bars = ax.bar(methods, estimates, color=colors, edgecolor="black", alpha=0.8,
                  yerr=errors_T, capsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ATE (Effect of SMS on Attendance)", fontsize=12)
    ax.set_title("Causal Effect of SMS Reminders on Attendance", fontsize=13)

    for bar, est in zip(bars, estimates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{est:.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "ate_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
