"""
ClinicSense – Deep Tabular Models & Statistical Significance Tests
Our ROC-AUC is around 0.64 which feels low, but is it actually meaningful?
This module adds:
  - TabNet (Arik & Pfister 2019) – attention-based deep tabular model
  - MLP baseline – to see if a neural net can beat tree ensembles
  - McNemar's test – are two classifiers statistically different?
  - DeLong's test – is the AUC difference significant?

The goal: scientifically show that no model does *meaningfully* better
on this dataset, which is an important result in itself.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings
from itertools import combinations

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy import stats

from src.config import FIGURES_DIR, RANDOM_STATE, CLASS_WEIGHT

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# --- TabNet (deep attention-based tabular model) ---
def build_tabnet(X_train, y_train, X_val=None, y_val=None,
                 max_epochs: int = 100, patience: int = 15):
    """
    Train a TabNet classifier for tabular data.

    Returns (model, history_dict) or (None, None) if pytorch-tabnet not installed.
    """
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier

        # Handle class imbalance via sample weights
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        weight_map = {0: 1.0, 1: n_neg / max(n_pos, 1)}
        sample_weights = np.array([weight_map[y] for y in y_train])

        model = TabNetClassifier(
            n_d=16, n_a=16,
            n_steps=4,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            seed=RANDOM_STATE,
            verbose=0,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=None,
        )

        X_tr = X_train.values if hasattr(X_train, 'values') else X_train
        y_tr = y_train.values if hasattr(y_train, 'values') else y_train

        fit_kwargs = {
            "X_train": X_tr,
            "y_train": y_tr,
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": 1024,
            "virtual_batch_size": 128,
            "weights": sample_weights,
        }

        if X_val is not None and y_val is not None:
            X_v = X_val.values if hasattr(X_val, 'values') else X_val
            y_v = y_val.values if hasattr(y_val, 'values') else y_val
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["eval_metric"] = ["auc"]

        model.fit(**fit_kwargs)
        history = model.history if hasattr(model, 'history') else {}
        return model, history

    except ImportError:
        print("[DeepTabular] pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
        return None, None
    except Exception as e:
        print(f"[DeepTabular] TabNet error: {e}")
        return None, None


def tabnet_predict_proba(model, X):
    """Get probability predictions from a TabNet model."""
    if model is None:
        return None
    X_arr = X.values if hasattr(X, 'values') else X
    return model.predict_proba(X_arr)[:, 1]


def tabnet_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract TabNet feature importances."""
    if model is None:
        return pd.DataFrame()
    importances = model.feature_importances_
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


# --- MLP model (sklearn, no GPU needed) ---
# I initially wanted FT-Transformer but it needs PyTorch + a lot of code,
# so I'm using a deep MLP as a simpler neural baseline.
def build_mlp_model() -> Pipeline:
    """
    Build an MLP classifier with feature scaling.
    Serves as a deep-learning baseline using sklearn (no GPU needed).
    Architecture: 256 → 128 → 64 with ReLU and dropout-like alpha regularisation.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,          # L2 regularisation (acts like dropout)
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            verbose=False,
        )),
    ])


# --- McNemar's test ---
# ref: Dietterich (1998) "Approximate Statistical Tests for Comparing
#      Supervised Classification Learning Algorithms"
def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray,
                 model_a_name: str = "A",
                 model_b_name: str = "B") -> dict:
    """
    McNemar's test: are two classifiers significantly different?

    Constructs a 2×2 contingency table:
                  B correct  B wrong
    A correct  [  n00,        n01   ]
    A wrong    [  n10,        n11   ]

    Tests H0: n01 == n10 (models make the same errors).

    Returns dict with chi2 statistic, p-value, and verdict.
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    # Contingency counts
    n00 = np.sum((correct_a == 1) & (correct_b == 1))  # both correct
    n01 = np.sum((correct_a == 1) & (correct_b == 0))  # A correct, B wrong
    n10 = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B correct
    n11 = np.sum((correct_a == 0) & (correct_b == 0))  # both wrong

    # McNemar test statistic (with continuity correction)
    if n01 + n10 == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n01_a_right_b_wrong": int(n01),
        "n10_a_wrong_b_right": int(n10),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "verdict": (f"{model_a_name} ≠ {model_b_name}" if p_value < 0.05
                    else f"No significant difference"),
    }


# --- DeLong's test (AUC comparison) ---
# ref: DeLong et al. (1988), Biometrics
def _delong_roc_variance(y_true, y_score):
    """Compute DeLong variance – this is the tricky maths part."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m = len(pos)
    n = len(neg)

    # placement values (Sun & Xu 2014 notation)
    v_pos = np.array([np.mean(y_score_p > neg) + 0.5 * np.mean(y_score_p == neg)
                      for y_score_p in pos])
    v_neg = np.array([np.mean(pos > y_score_n) + 0.5 * np.mean(pos == y_score_n)
                      for y_score_n in neg])

    auc = np.mean(v_pos)
    var = (np.var(v_pos, ddof=1) / m + np.var(v_neg, ddof=1) / n)
    return auc, var


def delong_test(y_true: np.ndarray,
                y_score_a: np.ndarray,
                y_score_b: np.ndarray,
                model_a_name: str = "A",
                model_b_name: str = "B") -> dict:
    """
    DeLong's test for comparing two ROC-AUC values.

    Tests H0: AUC_A == AUC_B.

    Returns dict with z-statistic, p-value, and AUC difference.
    """
    auc_a, var_a = _delong_roc_variance(y_true, y_score_a)
    auc_b, var_b = _delong_roc_variance(y_true, y_score_b)

    # Covariance (simplified: assume independent for robustness)
    diff = auc_a - auc_b
    se = np.sqrt(var_a + var_b)

    if se == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = diff / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(diff),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "verdict": (f"{model_a_name} AUC ≠ {model_b_name} AUC" if p_value < 0.05
                    else "No significant AUC difference"),
    }


# --- pairwise comparison of all models ---
def pairwise_model_tests(y_true: np.ndarray,
                         model_predictions: dict,
                         model_probabilities: dict) -> pd.DataFrame:
    """
    Run McNemar and DeLong tests for all model pairs.

    Parameters
    ----------
    y_true             : true labels
    model_predictions  : {model_name: y_pred}
    model_probabilities: {model_name: y_prob}

    Returns DataFrame with all pairwise test results.
    """
    records = []
    model_names = list(model_predictions.keys())

    for (a, b) in combinations(model_names, 2):
        mcnemar = mcnemar_test(y_true, model_predictions[a], model_predictions[b],
                               a, b)
        delong = delong_test(y_true, model_probabilities[a], model_probabilities[b],
                              a, b)
        records.append({
            "Model A": a,
            "Model B": b,
            "AUC A": delong["auc_a"],
            "AUC B": delong["auc_b"],
            "AUC Diff": delong["auc_diff"],
            "DeLong p": delong["p_value"],
            "DeLong Sig?": delong["significant_at_005"],
            "McNemar χ²": mcnemar["chi2"],
            "McNemar p": mcnemar["p_value"],
            "McNemar Sig?": mcnemar["significant_at_005"],
        })

    return pd.DataFrame(records)


# --- visualisation ---
def plot_model_comparison_with_tests(comparison_df: pd.DataFrame,
                                     pairwise_df: pd.DataFrame,
                                     save: bool = True):
    """
    Extended model comparison bar chart with statistical significance annotations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # AUC comparison
    if "roc_auc" in comparison_df.columns:
        auc_col = "roc_auc"
    elif "AUC A" in comparison_df.columns:
        auc_col = None  # use pairwise
    else:
        auc_col = None

    if auc_col:
        aucs = comparison_df[auc_col].sort_values(ascending=True)
        colors = ["#4CAF50" if v == aucs.max() else "steelblue" for v in aucs.values]
        aucs.plot(kind="barh", ax=axes[0], color=colors, edgecolor="black")
        axes[0].set_xlabel("ROC-AUC", fontsize=12)
        axes[0].set_title("Model ROC-AUC Comparison", fontsize=13)
        axes[0].grid(axis="x", alpha=0.3)

    # Significance heatmap
    if len(pairwise_df) > 0:
        models = sorted(set(pairwise_df["Model A"]).union(pairwise_df["Model B"]))
        n = len(models)
        p_matrix = np.ones((n, n))
        for _, row in pairwise_df.iterrows():
            i = models.index(row["Model A"])
            j = models.index(row["Model B"])
            p_matrix[i, j] = row["DeLong p"]
            p_matrix[j, i] = row["DeLong p"]

        mask = np.eye(n, dtype=bool)
        sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".3f",
                    xticklabels=models, yticklabels=models,
                    cmap="RdYlGn", vmin=0, vmax=0.1,
                    ax=axes[1], cbar_kws={"label": "p-value"})
        axes[1].set_title("DeLong Test p-values (AUC Comparison)", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "No pairwise tests available",
                     ha="center", va="center", fontsize=12)
        axes[1].set_title("Statistical Tests", fontsize=13)

    plt.suptitle("Deep Tabular Model Evaluation + Statistical Tests",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "deep_tabular_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tabnet_feature_importance(importance_df: pd.DataFrame,
                                   top_n: int = 15,
                                   save: bool = True):
    """Plot TabNet feature importance."""
    if importance_df is None or len(importance_df) == 0:
        return

    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="importance", y="feature",
                palette="magma", ax=ax)
    ax.set_title(f"TabNet – Top {top_n} Feature Importances", fontsize=13)
    ax.set_xlabel("Importance (attention weight)")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "tabnet_feature_importance.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
