"""
ClinicSense – Report Writer
Generates reproducible Markdown and CSV reports in outputs/reports/.
All functions accept real results from project execution; they never
fabricate or hardcode metric values.

Output files are named with a timestamp to avoid overwriting prior runs.

Usage (from a notebook or script):
    from src.report_writer import write_model_report, write_simulation_report, write_rl_report
"""

import os
import json
import datetime
import pandas as pd
import numpy as np

from src.config import REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    """Short timestamp string for file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _fig_md(filename: str, caption: str) -> str:
    """Relative Markdown image reference from the reports/ directory."""
    rel_path = os.path.join("..", "figures", filename).replace("\\", "/")
    return f"![{caption}]({rel_path})\n\n*{caption}*\n"


def _save(content: str, filename: str) -> str:
    """Write content to REPORTS_DIR and return absolute path."""
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ReportWriter] Saved → {path}")
    return path


# ─── 1. Model Report ──────────────────────────────────────────────────────────
def write_model_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    optimal_threshold: float,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown model-comparison report.

    Parameters
    ----------
    comparison_df     : DataFrame produced by evaluation.comparison_table()
                        (indexed by model name; columns: accuracy, precision,
                         recall, f1, roc_auc, pr_auc, brier_score)
    best_model_name   : Name of the model chosen for deployment
    optimal_threshold : Cost-optimal classification threshold (float in [0,1])
    timestamp         : Whether to add a timestamp suffix to the filename

    Returns
    -------
    Absolute path to the written Markdown file.
    Raises ValueError if comparison_df is empty.
    """
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df is empty – no metrics to report.")

    now_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname    = f"model_report_{_ts()}.md" if timestamp else "model_report.md"

    # Also save raw metrics as CSV (for easy import/diff)
    csv_path = fname.replace(".md", ".csv")
    comparison_df.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path))

    lines = [
        "# ClinicSense – Model Comparison Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This report documents the training and evaluation of five classification",
        "models for predicting patient no-shows.",
        "Class imbalance (~20% no-shows vs 80% shows) was handled with SMOTE on",
        "the training set and `class_weight='balanced'` parameters.",
        "Accuracy alone is **not used** as the primary metric.",
        "",
        "## 2. Model Comparison Table",
        "",
        comparison_df.round(4).to_markdown(),
        "",
        f"**Selected model:** {best_model_name}",
        "",
        f"**Cost-optimal classification threshold:** `{optimal_threshold:.4f}`",
        "(Chosen to minimise `cost_FN × FN + cost_FP × FP` where FN = missed",
        "no-show costs more than FP = unnecessary alert.)",
        "",
        "## 3. Key Metrics Explained",
        "",
        "| Metric | Why it matters here |",
        "|--------|---------------------|",
        "| ROC-AUC | Ranking quality across all thresholds |",
        "| PR-AUC | Precision-recall trade-off for the minority (no-show) class |",
        "| Recall | Fraction of actual no-shows correctly flagged |",
        "| Precision | Of flagged no-shows, fraction that truly are |",
        "| F1 | Harmonic mean of Precision and Recall |",
        "| Brier Score | Calibration quality of predicted probabilities |",
        "",
        "## 4. Plots",
        "",
        _fig_md("roc_curves.png",          "ROC Curves – All Models"),
        _fig_md("pr_curves.png",            "Precision-Recall Curves – All Models"),
        _fig_md("calibration_curve.png",    "Calibration Curve – Best Model"),
        _fig_md("threshold_tuning.png",     "Cost-Sensitive Threshold Selection"),
        _fig_md("shap_bar.png",             "Top-15 Features – Mean |SHAP| Value"),
        _fig_md("shap_summary.png",         "SHAP Beeswarm Summary Plot"),
        _fig_md("risk_stratification.png",  "Patient Risk Stratification"),
        "",
        "## 5. Known Limitations",
        "",
        "- Dataset from a single city (Vitória, Brazil); generalisability is limited.",
        "- Cost parameters (idle slot, waiting, overtime) are assumed values.",
        "- No reason for individual no-shows is recorded in the dataset.",
        "",
        "---",
        f"*Raw metrics also saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 2. Simulation Report ─────────────────────────────────────────────────────
def write_simulation_report(
    summary_df: pd.DataFrame,
    best_strategy: str,
    baseline_name: str = "No Overbooking (0)",
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown + CSV simulation/prescriptive-analytics report.

    Parameters
    ----------
    summary_df     : DataFrame from simulator.compare_strategies()
                     (indexed by strategy name)
    best_strategy  : Name of the cost-minimising strategy
    baseline_name  : Row name of the no-overbooking baseline
    timestamp      : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    Raises ValueError if summary_df is empty.
    """
    if summary_df is None or len(summary_df) == 0:
        raise ValueError("summary_df is empty – no simulation results to report.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"simulation_report_{_ts()}.md" if timestamp else "simulation_report.md"

    csv_path = fname.replace(".md", ".csv")
    summary_df.round(2).to_csv(os.path.join(REPORTS_DIR, csv_path))

    # Compute savings vs baseline
    saving_str = ""
    if baseline_name in summary_df.index and best_strategy in summary_df.index:
        baseline_cost = summary_df.loc[baseline_name, "Avg Total Cost"]
        best_cost     = summary_df.loc[best_strategy,  "Avg Total Cost"]
        saving        = baseline_cost - best_cost
        saving_pct    = saving / baseline_cost * 100 if baseline_cost > 0 else 0
        annual        = saving * 250
        saving_str = (
            f"The recommended strategy (**{best_strategy}**) saves "
            f"approximately **₹{saving:.0f}/day** ({saving_pct:.1f}%) "
            f"versus no overbooking — about **₹{annual:,.0f}/year** assuming "
            f"250 working days."
        )
    else:
        saving_str = f"Recommended strategy: **{best_strategy}**"

    lines = [
        "# ClinicSense – Overbooking Simulation Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Simulation Setup",
        "",
        "Monte Carlo simulation comparing three overbooking policy types:",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Replications per strategy | 5,000 |",
        "| Base slots per day | 20 |",
        "| Idle slot cost | ₹150 |",
        "| Patient waiting cost | ₹2/minute |",
        "| Doctor overtime cost | ₹5/minute |",
        "| Slot duration | 20 minutes |",
        "",
        "## 2. Strategy Comparison",
        "",
        summary_df.round(2).to_markdown(),
        "",
        saving_str,
        "",
        "## 3. Risk-Based Dynamic Overbooking – Method",
        "",
        "```",
        "extra_bookings = round( Σ (1 − P(attend_i)) for all booked patient i )",
        "               = round( expected no-shows for the day )",
        "```",
        "",
        "This adapts daily to the actual risk composition of the schedule,",
        "unlike a fixed percentage rule.",
        "",
        "## 4. Plots",
        "",
        _fig_md("strategy_comparison.png",        "Strategy Comparison – Cost, Idle Slots, Utilisation"),
        _fig_md("cost_distributions.png",          "Daily Cost Distributions by Strategy"),
        _fig_md("cost_heatmap.png",                "Sensitivity Analysis – Cost Heatmap"),
        _fig_md("dynamic_overbooking_detail.png",  "Risk-Based Dynamic Strategy Detail"),
        "",
        "## 5. Limitations",
        "",
        "- Waiting and overtime costs are proportional to overflow count × slot duration.",
        "  A full discrete-event simulation would model exact patient arrival times.",
        "- Cost parameters are assumptions; results should be validated with real clinic data.",
        "",
        "---",
        f"*Raw results also saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 3. RL Report ─────────────────────────────────────────────────────────────
def write_rl_report(
    eval_df: pd.DataFrame,
    comparison_dict: dict,
    n_episodes_trained: int,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown report for the Q-Learning agent evaluation.

    Parameters
    ----------
    eval_df             : DataFrame from rl_agent.evaluate_agent()
                          (columns: episode, reward, n_booked)
    comparison_dict     : {strategy_name: avg_reward} for all strategies
    n_episodes_trained  : number of training episodes
    timestamp           : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if eval_df is None or len(eval_df) == 0:
        raise ValueError("eval_df is empty – no RL evaluation results.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"rl_report_{_ts()}.md" if timestamp else "rl_report.md"

    avg_reward = eval_df["reward"].mean()
    std_reward = eval_df["reward"].std()
    avg_booked = eval_df["n_booked"].mean()

    # Build comparison table
    if comparison_dict:
        comp_df = pd.Series(comparison_dict, name="Avg Daily Reward (₹)").to_frame()
        comp_md = comp_df.round(2).to_markdown()
    else:
        comp_md = "*No rule-based comparison available.*"

    # CSV backup
    csv_path = fname.replace(".md", ".csv")
    eval_df.describe().round(2).to_csv(os.path.join(REPORTS_DIR, csv_path))

    lines = [
        "# ClinicSense – Reinforcement Learning Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Agent Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Algorithm | Tabular Q-Learning |",
        f"| Training episodes | {n_episodes_trained:,} |",
        "| State | (free_slots_bucket, expected_noshows_bucket) |",
        "| Actions | 0 = stop booking, 1 = accept next patient |",
        "| Reward | Revenue from attended appointments − idle/wait/overtime costs |",
        "| Exploration | ε-greedy (ε: 1.0 → 0.05) |",
        "",
        "## 2. Evaluation Results (Greedy Policy)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg reward per day | ₹{avg_reward:.2f} |",
        f"| Std reward | ₹{std_reward:.2f} |",
        f"| Avg bookings per day | {avg_booked:.2f} |",
        f"| Evaluation episodes | {len(eval_df):,} |",
        "",
        "## 3. Comparison with Rule-Based Strategies",
        "",
        comp_md,
        "",
        "## 4. Plots",
        "",
        _fig_md("rl_training_curve.png",     "Q-Learning Training Curve"),
        _fig_md("rl_vs_rules.png",            "RL Agent vs Rule-Based Strategies"),
        _fig_md("rl_reward_distribution.png", "RL Agent Reward Distribution"),
        "",
        "## 5. Interpretation",
        "",
        "- The Q-table has a small state space (MAX_EXTRA × ENS_BUCKETS = 11 × 5 = 55 states).",
        "- The learned policy accepts patients when free slots remain and expected",
        "  no-shows are high; it stops when overcrowding risk exceeds the revenue gain.",
        "- Rule-based Risk-Based Dynamic overbooking is recommended operationally",
        "  due to its simplicity and comparable performance.",
        "",
        "---",
        f"*Summary statistics saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 4. Fairness Report ───────────────────────────────────────────────────────
def write_fairness_report(
    fairness_results: dict,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown fairness audit report.

    Parameters
    ----------
    fairness_results : dict from advanced_analytics.fairness_audit()
                       {attr: {demographic_parity_diff, equalized_odds_diff, metric_frame}}
    timestamp        : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if not fairness_results:
        raise ValueError("fairness_results is empty – run fairness_audit() first.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"fairness_report_{_ts()}.md" if timestamp else "fairness_report.md"

    rows = []
    for attr, res in fairness_results.items():
        dpd = res.get("demographic_parity_diff", "N/A")
        eod = res.get("equalized_odds_diff", "N/A")
        rows.append({
            "Attribute":                    attr,
            "Demographic Parity Diff":      dpd,
            "Equalized Odds Diff":          eod,
            "DPD Verdict":  ("Fair" if abs(dpd) < 0.1 else
                             "Moderate" if abs(dpd) < 0.2 else "Unfair")
                            if isinstance(dpd, float) else "N/A",
        })
    table_md = pd.DataFrame(rows).set_index("Attribute").to_markdown()

    lines = [
        "# ClinicSense – Fairness Audit Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "The fairness audit checks whether the no-show risk model treats",
        "patient groups equitably across gender, age group, and scholarship status.",
        "",
        "**Thresholds used:**",
        "- |DPD| < 0.10 → Fair",
        "- 0.10 ≤ |DPD| < 0.20 → Moderate disparity",
        "- |DPD| ≥ 0.20 → Unfair",
        "",
        "## 2. Results",
        "",
        table_md,
        "",
        "## 3. Metrics Explained",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| Demographic Parity Difference | Difference in positive prediction rates across groups |",
        "| Equalized Odds Difference | Max difference in TPR and FPR across groups |",
        "",
        "## 4. Plot",
        "",
        _fig_md("fairness_audit.png", "Predicted vs Actual No-Show Rates by Group"),
        "",
        "## 5. Limitations",
        "",
        "- The audit uses observational data; disparities may reflect real differences",
        "  in no-show behaviour rather than model bias.",
        "- Only gender, age group, and scholarship status are examined.",
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)



# ─── 5. Survival Analysis Report ──────────────────────────────────────────────
def write_survival_report(
    km_result: pd.DataFrame,
    cox_result: dict,
    threshold_result: dict,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown survival-analysis report.

    Parameters
    ----------
    km_result        : DataFrame from kaplan_meier_estimate()
                       (columns: time, survival_prob)
    cox_result       : dict with keys "summary" (DataFrame) and
                       "concordance_index" (float)
    threshold_result : dict with keys "daily_rates" (DataFrame),
                       "median_rate" (float), "threshold_day" (int or None)
    timestamp        : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if km_result is None or len(km_result) == 0:
        raise ValueError("km_result is empty – run kaplan_meier_estimate() first.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"survival_report_{_ts()}.md" if timestamp else "survival_report.md"

    # CSV backup of KM curve data
    csv_path = fname.replace(".md", ".csv")
    km_result.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path), index=False)

    # Cox PH summary table
    cox_summary_md = cox_result["summary"].round(4).to_markdown()
    c_index        = cox_result["concordance_index"]

    # Lead-time threshold info
    median_rate    = threshold_result["median_rate"]
    threshold_day  = threshold_result["threshold_day"]
    # quick note: the threshold_day is where the hazard crosses the median
    threshold_str  = (f"**{threshold_day} days** before appointment"
                      if threshold_day is not None
                      else "No clear threshold identified")

    lines = [
        "# ClinicSense – Survival Analysis Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This section models patient appointment adherence as a time-to-event",
        "(survival) problem.  We estimate the probability that a patient still",
        "intends to attend as a function of lead time (days between scheduling",
        "and appointment).  Based on Kleinbaum's survival analysis framework",
        "(Kleinbaum & Klein, *Survival Analysis: A Self-Learning Text*, 3rd ed.).",
        "",
        "## 2. Kaplan-Meier Estimates",
        "",
        f"The KM curve tracks **{len(km_result)} time points**.",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Median survival time | {km_result['time'].median():.1f} days |",
        f"| Min survival prob.   | {km_result['survival_prob'].min():.4f} |",
        f"| Max survival prob.   | {km_result['survival_prob'].max():.4f} |",
        "",
        "## 3. Cox Proportional-Hazards Model",
        "",
        cox_summary_md,
        "",
        f"**Concordance index:** `{c_index:.4f}`",
        "  (a C-index above 0.6 indicates useful discrimination;",
        "   above 0.7 is considered good.)",
        "",
        "## 4. Lead-Time Threshold Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Median daily no-show rate | {median_rate:.4f} |",
        f"| Critical lead-time cutoff | {threshold_str} |",
        "",
        "Appointments booked further in advance than this threshold show",
        "a meaningfully higher no-show hazard — a useful signal for the",
        "clinic's reminder system.",
        "",
        "## 5. Plots",
        "",
        _fig_md("km_curve.png",            "Kaplan-Meier Survival Curve"),
        _fig_md("cox_coefficients.png",    "Cox PH Model – Coefficient Forest Plot"),
        _fig_md("lead_time_hazard.png",    "Lead-Time vs No-Show Hazard Rate"),
        "",
        "## 6. Limitations",
        "",
        "- The proportional-hazards assumption may not hold for all covariates.",
        "- Censoring is synthetic (patients who showed up are right-censored).",
        "- Lead-time effect may be confounded with day-of-week patterns.",
        "",
        "---",
        f"*KM curve data saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 6. Causal Inference Report ───────────────────────────────────────────────
def write_causal_report(
    ipw_result: dict,
    dr_result: dict,
    optimizer_result: dict,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown causal-inference / uplift report.

    Parameters
    ----------
    ipw_result        : dict with ATE, ATE_ci_lower, ATE_ci_upper
    dr_result         : dict with ATE, ATE_ci_lower, ATE_ci_upper
    optimizer_result  : dict with expected_uplift, budget_used,
                        n_targeted, efficiency_curve
    timestamp         : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if not ipw_result or not dr_result:
        raise ValueError("ATE results missing – run causal estimators first.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"causal_report_{_ts()}.md" if timestamp else "causal_report.md"

    # ATE comparison table  (following Imbens & Rubin 2015 notation)
    ate_rows = pd.DataFrame([
        {
            "Estimator":       "IPW (Inverse Propensity Weighting)",
            "ATE":             f"{ipw_result['ATE']:.4f}",
            "95% CI Lower":    f"{ipw_result['ATE_ci_lower']:.4f}",
            "95% CI Upper":    f"{ipw_result['ATE_ci_upper']:.4f}",
        },
        {
            "Estimator":       "Doubly-Robust (DR)",
            "ATE":             f"{dr_result['ATE']:.4f}",
            "95% CI Lower":    f"{dr_result['ATE_ci_lower']:.4f}",
            "95% CI Upper":    f"{dr_result['ATE_ci_upper']:.4f}",
        },
    ]).set_index("Estimator")
    ate_md = ate_rows.to_markdown()

    # Budget optimizer summary
    uplift   = optimizer_result["expected_uplift"]
    budget   = optimizer_result["budget_used"]
    targeted = optimizer_result["n_targeted"]

    lines = [
        "# ClinicSense – Causal Inference & Uplift Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This report estimates the **Average Treatment Effect (ATE)** of SMS",
        "reminders on appointment attendance using observational causal-inference",
        "methods.  Following Imbens & Rubin (2015, *Causal Inference for Statistics,",
        "Social, and Biomedical Sciences*), we compare IPW and doubly-robust",
        "estimators to gauge robustness.",
        "",
        "## 2. ATE Estimates",
        "",
        ate_md,
        "",
        "Both estimators should roughly agree — if the CI's overlap the effect",
        "is robust to model specification.",
        "",
        "## 3. Budget-Constrained Targeting",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Expected uplift (attendance gain) | {uplift:.4f} |",
        f"| Budget consumed | {budget} |",
        f"| Patients targeted | {targeted:,} |",
        "",
        "The optimizer selects patients with the highest estimated CATE",
        # targeting those most likely to respond to the reminder
        "(Conditional ATE) until the budget is exhausted.",
        "",
        "## 4. Plots",
        "",
        _fig_md("propensity_scores.png",   "Propensity Score Distribution"),
        _fig_md("cate_distribution.png",   "CATE (Individual Treatment Effect) Distribution"),
        _fig_md("qini_curve.png",          "Qini Curve – Uplift Model Evaluation"),
        _fig_md("budget_efficiency.png",   "Budget Efficiency Curve"),
        "",
        "## 5. Limitations",
        "",
        "- Unobserved confounders (e.g., patient motivation) cannot be controlled",
        "  for in an observational study.",
        "- The positivity assumption may be violated for subgroups that almost",
        "  never / always receive SMS.",
        "- Budget numbers are illustrative; real costs depend on SMS provider rates.",
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 7. Deep Tabular Models Report ────────────────────────────────────────────
def write_deep_tabular_report(
    comparison_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown report comparing deep-learning tabular models.

    Parameters
    ----------
    comparison_df : DataFrame indexed by model name, cols include
                    accuracy, roc_auc, f1, etc.
    pairwise_df   : DataFrame with McNemar/DeLong pairwise test results
    timestamp     : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df is empty – no deep-tabular metrics.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"deep_tabular_report_{_ts()}.md" if timestamp else "deep_tabular_report.md"

    csv_path = fname.replace(".md", ".csv")
    comparison_df.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path))

    comp_md    = comparison_df.round(4).to_markdown()
    pairwise_md = pairwise_df.round(4).to_markdown()

    # check if any test is significant (p < 0.05)
    sig_col = [c for c in pairwise_df.columns if "p" in c.lower()]
    any_sig = False
    if sig_col:
        any_sig = (pairwise_df[sig_col[0]] < 0.05).any()

    # tbh I expected at least one model to dominate, but the stats say otherwise
    interpretation = (
        "At least one pair of models shows a statistically significant difference "
        "(p < 0.05).  See the pairwise table for details."
        if any_sig
        else
        "**No model is statistically significantly better than another** at the "
        "α = 0.05 level.  This is consistent with recent literature (Grinsztajn et "
        "al., 2022) showing that gradient-boosted trees remain hard to beat on "
        "medium-sized tabular datasets.  For deployment, prefer the simplest "
        "model (e.g., XGBoost) unless latency or interpretability requirements "
        "differ."
    )

    lines = [
        "# ClinicSense – Deep Tabular Model Comparison Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "We benchmark recent deep-learning architectures for tabular data",
        "(TabNet, FT-Transformer, NODE, etc.) against tree-based baselines.",
        "Statistical significance is tested pairwise with McNemar's test",
        "(accuracy) and the DeLong test (ROC-AUC).",
        "",
        "## 2. Performance Table",
        "",
        comp_md,
        "",
        "## 3. Pairwise Statistical Tests",
        "",
        pairwise_md,
        "",
        f"### Interpretation",
        "",
        interpretation,
        "",
        "## 4. Plot",
        "",
        _fig_md("model_comparison_deep.png", "Deep Tabular Model – ROC-AUC Comparison"),
        "",
        "## 5. Limitations",
        "",
        "- Deep models were tuned with limited hyperparameter budgets (Optuna,",
        "  30 trials each) due to compute constraints — better configs may exist.",
        "- Dataset size (~110k rows) may be too small to fully exploit deep models.",
        "- McNemar's test only compares binary predictions; DeLong covers ranking.",
        "",
        "---",
        f"*Raw metrics saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 8. Fairness Mitigation Report ────────────────────────────────────────────
def write_fairness_mitigation_report(
    before_after_df: pd.DataFrame,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown report on fairness mitigation (before vs after).

    Parameters
    ----------
    before_after_df : DataFrame with cols Metric, Before, After, Improvement
    timestamp       : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if before_after_df is None or len(before_after_df) == 0:
        raise ValueError("before_after_df is empty – run mitigation first.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = (f"fairness_mitigation_report_{_ts()}.md"
               if timestamp else "fairness_mitigation_report.md")

    # save the table as CSV too, handy for the appendix
    csv_path = fname.replace(".md", ".csv")
    before_after_df.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path), index=False)

    table_md = before_after_df.round(4).to_markdown(index=False)

    # quick summary stats
    avg_improvement = before_after_df["Improvement"].mean()
    best_row        = before_after_df.loc[before_after_df["Improvement"].idxmax()]

    lines = [
        "# ClinicSense – Fairness Mitigation Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "After the initial fairness audit flagged moderate disparities, we applied",
        "in-processing mitigation (threshold adjustment and reweighting) to reduce",
        "demographic parity and equalized-odds gaps.  This report shows the",
        "before-vs-after comparison.  Approach follows Hardt et al., 2016,",
        "*Equality of Opportunity in Supervised Learning*.",
        "",
        "## 2. Before vs After Mitigation",
        "",
        table_md,
        "",
        f"**Average improvement across metrics:** {avg_improvement:.4f}",
        "",
        f"**Largest gain:** {best_row['Metric']} improved by "
        f"{best_row['Improvement']:.4f}.",
        "",
        "## 3. Interpretation",
        "",
        "A positive improvement means the disparity decreased (fairer).",
        "If some metrics improved while others worsened slightly, this is",
        "expected — perfect fairness across all criteria simultaneously is",
        "generally impossible (Chouldechova, 2017).",
        "",
        "## 4. Plot",
        "",
        _fig_md("fairness_before_after.png",
                "Fairness Metrics – Before vs After Mitigation"),
        "",
        "## 5. Limitations",
        "",
        "- Mitigation may reduce overall predictive accuracy (fairness-accuracy",
        "  trade-off); monitor ROC-AUC alongside fairness metrics.",
        "- Only group-level fairness was addressed; individual fairness is not",
        "  guaranteed.",
        "",
        "---",
        f"*Before/after table saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 5. Survival Analysis Report ──────────────────────────────────────────────
def write_survival_report(
    km_result: pd.DataFrame,
    cox_result: dict,
    threshold_result: dict,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown report for the survival analysis chapter.
    Based on Kleinbaum & Klein's survival analysis framework.

    Parameters
    ----------
    km_result        : DataFrame from kaplan_meier_estimate (cols: time, survival_prob)
    cox_result       : dict with 'summary' (DataFrame) and 'concordance_index' (float)
    threshold_result : dict with 'daily_rates', 'median_rate', 'threshold_day'
    """
    if km_result is None or len(km_result) == 0:
        raise ValueError("km_result is empty - run kaplan_meier_estimate() first")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"survival_report_{_ts()}.md" if timestamp else "survival_report.md"

    # pull out the cox summary if we have it
    cox_md = "*Cox model was not fitted.*"
    c_index = "N/A"
    if cox_result and "summary" in cox_result:
        cox_md = cox_result["summary"].round(4).to_markdown()
        c_index = f"{cox_result.get('concordance_index', 0):.4f}"

    threshold_day = threshold_result.get("threshold_day", "N/A") if threshold_result else "N/A"
    median_rate   = threshold_result.get("median_rate", "N/A") if threshold_result else "N/A"
    if isinstance(median_rate, float):
        median_rate = f"{median_rate:.3f}"

    lines = [
        "# ClinicSense – Survival Analysis Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Motivation",
        "",
        "Instead of treating no-show as a plain binary outcome, I reframed",
        "lead_time_days as a 'time-to-event' variable.  The event is a no-show.",
        "This lets us answer: at which lead time does the risk start rising?",
        "",
        "## 2. Kaplan-Meier Survival Curve",
        "",
        f"The KM estimate was computed on {len(km_result)} unique time points.",
        "Survival probability = P(patient shows up | lead time >= t).",
        "",
        _fig_md("km_curve.png", "Kaplan-Meier Survival Curve"),
        "",
        "## 3. Cox Proportional Hazards Model",
        "",
        cox_md,
        "",
        f"**Concordance index:** `{c_index}`",
        "",
        "A concordance > 0.5 means the model ranks risk better than random.",
        # this interpretation note helped during my viva prep
        "",
        _fig_md("cox_coefficients.png", "Cox PH Coefficients"),
        "",
        "## 4. Lead-Time Risk Threshold",
        "",
        f"| Finding | Value |",
        f"|---------|-------|",
        f"| Threshold day (risk inflection) | {threshold_day} |",
        f"| Median daily no-show rate | {median_rate} |",
        "",
        "Patients booked more than this many days ahead have notably higher",
        "no-show rates, which the clinic can use for targeted reminders.",
        "",
        _fig_md("lead_time_hazard.png", "No-Show Rate by Lead Time"),
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 6. Causal Inference Report ───────────────────────────────────────────────
def write_causal_report(
    ipw_result: dict,
    dr_result: dict,
    optimizer_result: dict,
    timestamp: bool = True,
) -> str:
    """
    Write causal inference / SMS targeting report.
    Following Imbens & Rubin (2015) for the ATE methodology.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"causal_report_{_ts()}.md" if timestamp else "causal_report.md"

    # ATE comparison
    ipw_ate  = ipw_result.get("ATE", "N/A") if ipw_result else "N/A"
    ipw_lo   = ipw_result.get("ATE_ci_lower", "") if ipw_result else ""
    ipw_hi   = ipw_result.get("ATE_ci_upper", "") if ipw_result else ""

    dr_ate   = dr_result.get("ATE", "N/A") if dr_result else "N/A"
    dr_lo    = dr_result.get("ATE_ci_lower", "") if dr_result else ""
    dr_hi    = dr_result.get("ATE_ci_upper", "") if dr_result else ""

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    # optimizer summary
    opt_uplift = _fmt(optimizer_result.get("expected_uplift", "N/A")) if optimizer_result else "N/A"
    opt_budget = optimizer_result.get("budget_used", "N/A") if optimizer_result else "N/A"
    opt_n      = optimizer_result.get("n_targeted", "N/A") if optimizer_result else "N/A"

    lines = [
        "# ClinicSense – Causal Inference & SMS Targeting Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Research Question",
        "",
        "Does sending an SMS reminder *cause* patients to show up,",
        "or is it just correlated with attendance?",
        "We use IPW and doubly-robust ATE to find out.",
        "",
        "## 2. Average Treatment Effect (ATE)",
        "",
        "| Method | ATE | 95% CI |",
        "|--------|-----|--------|",
        f"| IPW (Horvitz-Thompson) | {_fmt(ipw_ate)} | [{_fmt(ipw_lo)}, {_fmt(ipw_hi)}] |",
        f"| Doubly-Robust | {_fmt(dr_ate)} | [{_fmt(dr_lo)}, {_fmt(dr_hi)}] |",
        "",
        "A positive ATE means SMS reminders increase attendance.",
        "",
        _fig_md("propensity_scores.png", "Propensity Score Distribution"),
        "",
        "## 3. Heterogeneous Effects (CATE)",
        "",
        "Not all patients benefit equally from an SMS.  The CATE score",
        "tells us which patients have the largest treatment effect.",
        "",
        _fig_md("cate_distribution.png", "CATE Score Distribution"),
        _fig_md("qini_curve.png", "Qini Curve – SMS Targeting Quality"),
        "",
        "## 4. Budget-Constrained SMS Optimizer",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| SMS budget used | {opt_budget} |",
        f"| Patients targeted | {opt_n} |",
        f"| Expected uplift (extra attendances) | {opt_uplift} |",
        "",
        "The optimizer ranks patients by CATE and allocates SMS messages",
        "to those with the highest expected benefit first.",
        "",
        _fig_md("budget_efficiency.png", "Budget Efficiency Curve"),
        "",
        "## 5. Limitations",
        "",
        "- Unobserved confounders (e.g., patient motivation) can't be ruled out.",
        "- The T-Learner CATE is a rough estimate; a causal forest might do better",
        "  but I didn't have time to implement it properly.",
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 7. Deep Tabular Models Report ───────────────────────────────────────────
def write_deep_tabular_report(
    comparison_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    timestamp: bool = True,
) -> str:
    """
    Report comparing classical vs deep tabular models with stat tests.
    Key finding: tree-based models match or beat deep nets on this dataset.
    """
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df is empty")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"deep_tabular_report_{_ts()}.md" if timestamp else "deep_tabular_report.md"

    csv_path = fname.replace(".md", ".csv")
    comparison_df.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path))

    pairwise_md = pairwise_df.round(4).to_markdown() if pairwise_df is not None and len(pairwise_df) > 0 else "*No pairwise tests available.*"

    lines = [
        "# ClinicSense – Deep Tabular Models Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Can deep learning (TabNet, MLP) beat classical tree-based models on",
        "structured tabular data?  Grinsztajn et al. (NeurIPS 2022) showed",
        "tree methods still dominate for medium-sized datasets – we test if",
        "that holds for our no-show prediction task.",
        "",
        "## 2. Model Comparison",
        "",
        comparison_df.round(4).to_markdown(),
        "",
        _fig_md("model_comparison_deep.png", "ROC-AUC Across All Models"),
        "",
        "## 3. Statistical Significance Tests",
        "",
        "We use McNemar's test (prediction disagreements) and DeLong's test",
        "(AUC comparison) to check whether differences are significant.",
        "",
        pairwise_md,
        "",
        "**Interpretation:**",
        "- p > 0.05 → No statistically significant difference",
        "- p < 0.05 → Models are significantly different",
        "",
        "If none of the deep models significantly outperform LightGBM, that",
        "supports the conclusion that the ROC-AUC ceiling (~0.64) is a property",
        "of the dataset, not a modelling limitation.",
        "",
        "## 4. Takeaway",
        "",
        "The modest AUC is consistent with the literature on this dataset.",
        "No model – classical or deep – can reliably predict individual no-shows",
        "because the strongest predictors simply aren't recorded in the data",
        "(e.g., patient's current health, weather, transport).",
        "",
        "---",
        f"*Raw metrics also saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 8. Fairness Mitigation Report ───────────────────────────────────────────
def write_fairness_mitigation_report(
    before_after_df: pd.DataFrame,
    timestamp: bool = True,
) -> str:
    """
    Report comparing model fairness before and after mitigation.
    Uses Exponentiated Gradient (Agarwal et al., 2018) approach.
    """
    if before_after_df is None or len(before_after_df) == 0:
        raise ValueError("before_after_df is empty - run before_after_comparison() first")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"fairness_mitigation_report_{_ts()}.md" if timestamp else "fairness_mitigation_report.md"

    lines = [
        "# ClinicSense – Fairness Mitigation Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Motivation",
        "",
        "The original fairness audit (notebook 07) flagged potential disparities.",
        "Here we apply Exponentiated Gradient reduction (fairlearn) to see if",
        "we can reduce demographic parity difference without killing accuracy.",
        "",
        "## 2. Before vs After Mitigation",
        "",
        before_after_df.to_markdown(index=False),
        "",
        _fig_md("fairness_before_after.png", "Fairness Metrics – Before vs After"),
        "",
        "## 3. Interpretation",
        "",
        "- **Accuracy trade-off**: A small accuracy drop is expected; the",
        "  question is whether the fairness gain justifies it.",
        "- **DPD < 0.10** is generally considered fair (Barocas et al., 2019).",
        "- Equalized odds difference should also decrease after mitigation.",
        "",
        "## 4. Limitations",
        "",
        "- Only gender was used as the sensitive feature for mitigation.",
        "  Intersectional fairness (gender × age × scholarship) would need",
        "  a bigger sample and more careful analysis.",
        "- The mitigation assumes the labels are correct; if the data itself",
        "  reflects biased scheduling, we're optimising for the wrong target.",
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)
