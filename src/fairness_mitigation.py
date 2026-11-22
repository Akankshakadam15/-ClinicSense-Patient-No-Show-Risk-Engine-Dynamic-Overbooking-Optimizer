"""
ClinicSense – Fairness Mitigation
Notebook 07 showed the audit results; this module actually *fixes* the
disparity.  I use Exponentiated Gradient from fairlearn (Agarwal et al. 2018)
because it wraps any sklearn estimator and directly minimises the
demographic parity gap during training.

Also tried ThresholdOptimizer (post-processing) as a lighter alternative.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)

from src.config import FIGURES_DIR, RANDOM_STATE

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# --- exponentiated gradient (in-processing) ---
def mitigate_exponentiated_gradient(estimator,
                                     X_train,
                                     y_train,
                                     sensitive_train: pd.Series,
                                     constraint: str = "demographic_parity",
                                     eps: float = 0.01):
    """
    Apply fairlearn's Exponentiated Gradient to produce a fair classifier.

    Parameters
    ----------
    estimator        : sklearn-compatible base estimator (unfitted)
    X_train          : training features
    y_train          : training labels
    sensitive_train  : sensitive attribute for training data
    constraint       : 'demographic_parity' or 'equalized_odds'
    eps              : fairness tolerance (smaller = stricter)

    Returns
    -------
    mitigated_model : fitted ExponentiatedGradient model (or None on error)
    """
    try:
        from fairlearn.reductions import (
            ExponentiatedGradient,
            DemographicParity,
            EqualizedOdds,
        )

        if constraint == "demographic_parity":
            constr = DemographicParity(difference_bound=eps)
        elif constraint == "equalized_odds":
            constr = EqualizedOdds(difference_bound=eps)
        else:
            raise ValueError(f"Unknown constraint: {constraint}")

        mitigated = ExponentiatedGradient(
            estimator=estimator,
            constraints=constr,
            max_iter=50,
        )
        mitigated.fit(X_train, y_train, sensitive_features=sensitive_train)
        return mitigated

    except ImportError:
        print("[Fairness] fairlearn not installed. Run: pip install fairlearn")
        return None
    except Exception as e:
        print(f"[Fairness Mitigation] Error: {e}")
        return None


# --- threshold optimizer (post-processing alternative) ---
# this is simpler – just adjusts the decision threshold per group
def mitigate_threshold_optimizer(estimator,
                                  X_test,
                                  y_test,
                                  sensitive_test: pd.Series,
                                  constraint: str = "demographic_parity"):
    """
    Apply fairlearn's ThresholdOptimizer for post-processing fairness.

    Parameters
    ----------
    estimator       : pre-fitted classifier
    X_test          : test features
    y_test          : test labels
    sensitive_test  : sensitive attribute for test data
    constraint      : 'demographic_parity' or 'equalized_odds'

    Returns
    -------
    postprocessed_model : fitted ThresholdOptimizer (or None on error)
    """
    try:
        from fairlearn.postprocessing import ThresholdOptimizer

        if constraint == "demographic_parity":
            constr = "demographic_parity"
        else:
            constr = "equalized_odds"

        to = ThresholdOptimizer(
            estimator=estimator,
            constraints=constr,
            prefit=True,
        )
        to.fit(X_test, y_test, sensitive_features=sensitive_test)
        return to

    except ImportError:
        print("[Fairness] fairlearn not installed.")
        return None
    except Exception as e:
        print(f"[Threshold Optimizer] Error: {e}")
        return None


# --- fairness metrics helper ---
def compute_fairness_metrics(y_true, y_pred, sensitive_features: pd.Series,
                              label: str = "Model") -> dict:
    """
    Compute key fairness metrics for a single model.

    Returns dict with:
      - overall accuracy, F1
      - demographic_parity_diff
      - equalized_odds_diff
      - per-group accuracy, selection rate
    """
    try:
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference,
            equalized_odds_difference, selection_rate,
        )

        mf = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        dpd = demographic_parity_difference(y_true, y_pred,
                                             sensitive_features=sensitive_features)
        eod = equalized_odds_difference(y_true, y_pred,
                                         sensitive_features=sensitive_features)

        return {
            "label": label,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "demographic_parity_diff": float(dpd),
            "equalized_odds_diff": float(eod),
            "group_metrics": mf.by_group.to_dict(),
            "metric_frame": mf,
        }

    except ImportError:
        return {
            "label": label,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "demographic_parity_diff": None,
            "equalized_odds_diff": None,
        }


def before_after_comparison(y_true,
                            y_pred_before,
                            y_pred_after,
                            sensitive_features: pd.Series) -> pd.DataFrame:
    """
    Compare fairness metrics before and after mitigation.

    Returns DataFrame with columns: metric, before, after, improvement.
    """
    before = compute_fairness_metrics(y_true, y_pred_before, sensitive_features,
                                       label="Before")
    after = compute_fairness_metrics(y_true, y_pred_after, sensitive_features,
                                      label="After")

    records = []
    for metric in ["accuracy", "f1", "demographic_parity_diff", "equalized_odds_diff"]:
        b = before.get(metric)
        a = after.get(metric)
        if b is not None and a is not None:
            if metric in ("demographic_parity_diff", "equalized_odds_diff"):
                improvement = abs(b) - abs(a)  # reduction in disparity = good
            else:
                improvement = a - b  # increase in performance = good
            records.append({
                "Metric": metric,
                "Before": round(b, 4),
                "After": round(a, 4),
                "Improvement": round(improvement, 4),
            })

    return pd.DataFrame(records)


# --- plots ---
def plot_fairness_before_after(comparison_df: pd.DataFrame,
                                save: bool = True):
    """Bar chart comparing fairness metrics before and after mitigation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Performance metrics
    perf_metrics = comparison_df[comparison_df["Metric"].isin(["accuracy", "f1"])]
    if len(perf_metrics) > 0:
        x = np.arange(len(perf_metrics))
        w = 0.35
        axes[0].bar(x - w/2, perf_metrics["Before"], w, label="Before",
                     color="steelblue", edgecolor="black")
        axes[0].bar(x + w/2, perf_metrics["After"], w, label="After",
                     color="darkorange", edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(perf_metrics["Metric"])
        axes[0].set_ylabel("Score")
        axes[0].set_title("Performance Metrics", fontsize=13)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis="y", alpha=0.3)

    # Fairness metrics
    fair_metrics = comparison_df[comparison_df["Metric"].isin(
        ["demographic_parity_diff", "equalized_odds_diff"]
    )]
    if len(fair_metrics) > 0:
        x = np.arange(len(fair_metrics))
        axes[1].bar(x - w/2, fair_metrics["Before"].abs(), w, label="Before",
                     color="#F44336", edgecolor="black")
        axes[1].bar(x + w/2, fair_metrics["After"].abs(), w, label="After",
                     color="#4CAF50", edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["DPD", "EOD"])
        axes[1].set_ylabel("|Disparity|")
        axes[1].set_title("Fairness Metrics (lower = fairer)", fontsize=13)
        axes[1].legend()
        axes[1].axhline(0.1, color="grey", linestyle="--", linewidth=1,
                         label="Fair threshold")
        axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Fairness Mitigation – Before vs After",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "fairness_before_after.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_selection_rates(before_metrics: dict,
                                after_metrics: dict,
                                save: bool = True):
    """Compare per-group selection rates before and after mitigation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (metrics, label) in enumerate(
        [(before_metrics, "Before"), (after_metrics, "After")]
    ):
        mf = metrics.get("metric_frame")
        if mf is not None:
            by_group = mf.by_group
            if "selection_rate" in by_group.columns:
                by_group["selection_rate"].plot(
                    kind="bar", ax=axes[idx],
                    color="steelblue" if idx == 0 else "darkorange",
                    edgecolor="black"
                )
                axes[idx].set_title(f"Selection Rate by Group ({label})", fontsize=12)
                axes[idx].set_ylabel("Selection Rate")
                axes[idx].set_ylim(0, 1)
                axes[idx].grid(axis="y", alpha=0.3)
                axes[idx].tick_params(axis="x", rotation=0)

    plt.suptitle("Per-Group Selection Rates", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "group_selection_rates.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
