# ClinicSense – Viva Preparation Guide

**Student:** Kadam Akanksha Dinkar | **B.Sc. Data Science Honours**  
**Project:** ClinicSense: Patient No-Show Risk Engine & Dynamic Overbooking Optimizer

---

## 1. Problem and Motivation

**Q: What real-world problem does ClinicSense solve?**  
Outpatient clinics lose ~20% of booked appointments to no-shows. Every missed slot wastes doctor time, cuts revenue, and delays care. Clinics either ignore the problem or use a fixed overbooking rule that ignores individual patient risk.

**Q: Why is this a Predictive Analytics project, not just an ML project?**  
PA goes beyond building a model. ClinicSense covers: data collection → EDA → preprocessing → feature engineering → modelling → evaluation → **interpretation** (SHAP, risk labels) → **decision support** (simulator, policy recommendation). The prescriptive overbooking simulator is the key differentiator.

---

## 2. Dataset

**Q: What dataset is used?**  
Kaggle "Medical Appointment No Shows" – public clinics in Vitória, Brazil. ~110,000 rows, 14 columns, ~20% no-shows (imbalanced).

**Q: What data quality issues were found?**  
Negative ages, appointments scheduled after the visit date, and duplicate appointment IDs. All removed in `data_loader.py`.

**Q: What is the target variable?**  
`no_show`: 1 if patient missed the appointment, 0 if they attended.

---

## 3. Feature Engineering

**Q: What features did you engineer and why?**  

| Feature | Rationale |
|---------|-----------|
| `lead_time_days` | Longer wait → higher no-show risk (confirmed by Mann-Whitney test) |
| `prior_no_show_rate` | Past behaviour strongly predicts future (patient history) |
| `prior_appointments` | First-timers have less commitment |
| `age_group_enc` | Non-linear relationship with no-show rate |
| `appt_weekday`, `is_monday`, `is_friday` | Weekday no-show patterns exist |
| `condition_count` | Patients with chronic conditions attend more reliably |
| `same_day_booking` | Same-day bookings almost never result in no-shows |
| `neighbourhood_freq` | Frequency encoding for high-cardinality categorical |

**Q: How did you handle the neighbourhood variable?**  
Frequency encoding (replaces each neighbourhood with its proportion of total appointments). This preserves information without creating 81 dummy variables.

**Q: What is VIF and why did you compute it?**  
Variance Inflation Factor detects multicollinearity. VIF > 10 indicates a feature can be predicted from others, making model coefficients unstable. All features had VIF ≤ 10, so no removal was needed.

---

## 4. Handling Class Imbalance

**Q: Why is the dataset imbalanced and how did you handle it?**  
~80% show / ~20% no-show. If ignored, a model that always predicts "show" gets 80% accuracy but zero utility.

Approaches used:
- **SMOTE** on training data only (never on test data — data leakage prevention)
- **`class_weight='balanced'`** in LR, RF, LightGBM
- **`scale_pos_weight`** in XGBoost
- **Evaluation without accuracy** — used Recall, F1, ROC-AUC, PR-AUC, Brier Score

**Q: Why SMOTE only on training data?**  
Applying SMOTE to test data would create synthetic test samples that don't represent real patients, giving an optimistic and misleading evaluation.

---

## 5. Models and Comparison

**Q: Why five models?**  
Logistic Regression as the interpretable baseline; RF, XGBoost, LightGBM as strong tree ensembles; Stacking as the most complex ensemble. This gives a spectrum from interpretable to powerful.

**Q: Which model performed best and why?**  
LightGBM (tuned + calibrated) achieved the best balance of ROC-AUC and PR-AUC. Tree ensembles handle non-linear feature interactions (e.g., lead_time × prior_no_show_rate) that logistic regression misses.

**Q: What is probability calibration and why did you use it?**  
Raw classifier probabilities may not reflect true likelihoods (e.g., a model might predict 0.8 when the true rate is only 0.6). Isotonic calibration (via MAPIE SplitConformalClassifier or sklearn CalibratedClassifierCV) adjusts probabilities to match observed frequencies, which is critical for the overbooking simulator.

**Q: What is a cost-sensitive threshold?**  
Instead of 0.5, the threshold is chosen to minimise `cost_FN × FN + cost_FP × FP`. Missing a no-show (FN) causes an idle slot costing ₹150; a false alarm (FP) causes a minor admin overhead (~₹20). The optimal threshold is lower than 0.5, catching more no-shows at the cost of more false alarms.

---

## 6. SHAP Interpretability

**Q: What is SHAP and how did you use it?**  
SHapley Additive exPlanations assign each feature a contribution to each individual prediction. The TreeExplainer is efficient for gradient boosting models. Used to:
1. Rank global feature importance (mean |SHAP|)
2. Show dependence plots (how feature value affects prediction)
3. Explain individual high-risk predictions with waterfall charts

**Q: What were the top no-show predictors?**  
Lead time, prior no-show rate, prior appointments, SMS reminder, age group, and same-day booking flag — broadly matching domain knowledge and literature.

---

## 7. Overbooking Simulator

**Q: How does the Monte Carlo simulator work?**  
For each simulated day:
1. Sample a pool of patients with predicted attendance probabilities
2. Apply an overbooking strategy to decide how many to book
3. Simulate arrivals: each patient attends with probability `attend_prob_i`
4. Compute idle slots, overflow, and total cost
5. Repeat 5,000 times and average

**Q: What three strategies were compared?**  
1. No overbooking (baseline)
2. Fixed overbooking (+N extra patients, same every day)
3. Risk-based dynamic: extra_bookings = round(Σ(1 - attend_prob_i))

**Q: Why is risk-based dynamic better?**  
It adapts to the actual risk composition of each day's schedule. On a day with many high-risk patients, it books more extras. On a day with mostly reliable patients, it books fewer — avoiding unnecessary overcrowding.

**Q: What are the cost assumptions?**  
Idle slot: ₹150, patient waiting: ₹2/min, doctor overtime: ₹5/min, slot duration: 20 min. These are stated assumptions since the dataset has no real cost data.

---

## 8. Advanced Analytics

**Q: What is conformal prediction (MAPIE)?**  
Instead of a point probability, conformal prediction provides a prediction *set* with guaranteed coverage. At 90% confidence, at least 90% of test predictions contain the true label. Used here with MAPIE 1.6's `SplitConformalClassifier`.

**Q: What is the T-Learner uplift model?**  
Two separate models: one trained on SMS-sent patients, one on no-SMS patients. The difference in predicted attendance probability is the individual treatment effect (uplift). Patients with high positive uplift benefit most from being sent an SMS reminder.

**Q: What is PSI and what did it show?**  
Population Stability Index measures distributional drift between training and test periods. PSI < 0.1 = stable. The analysis confirmed all key features remained stable across the time split, meaning the model is robust over time.

**Q: What did the fairness audit find?**  
Demographic Parity Difference and Equalized Odds Difference were computed across gender, age group, and scholarship status using Fairlearn. Values < 0.1 indicate the model treats groups equitably.

---

## 9. Reinforcement Learning

**Q: What is the RL setup?**  
- **State:** (free_slots_bucket, expected_no-shows_bucket) — 55 discrete states
- **Action:** 0 = stop accepting bookings, 1 = accept next patient
- **Reward:** Revenue from attended appointments minus idle/wait/overtime costs
- **Algorithm:** Tabular Q-Learning with ε-greedy exploration

**Q: How does the RL agent compare to rule-based strategies?**  
After 3,000 episodes, the Q-Learning agent learns a policy competitive with risk-based dynamic overbooking. The rule-based strategy is still recommended operationally due to its simplicity and ex
plainability.

---

## 10. Survival Analysis (NB09)

**Q: Why reframe lead_time_days as a survival variable?**  
Standard classification treats lead time as just another numeric input. But by treating "time until no-show event" as a survival variable, we can estimate *when* risk changes — not just *whether* a patient is high-risk. This directly answers: "beyond how many days of lead time does the no-show rate spike?"

**Q: What is a Kaplan-Meier curve?**  
It's a non-parametric estimator of the survival function S(t) = P(patient attends | lead time ≥ t). The curve shows the probability of a patient still "surviving" (attending) as lead time increases. Each step down represents observed no-shows at that time point. I used the lifelines library for this. Reference: Kleinbaum & Klein, *Survival Analysis* (3rd ed.), Chapters 1-2.

**Q: What is the Nelson-Aalen estimator and how does it differ from KM?**  
Nelson-Aalen estimates the cumulative hazard function H(t) instead of the survival function. It's computed as the cumulative sum of d_i/n_i (events over at-risk count at each time). It tends to be more robust in small samples and is related to KM via S(t) ≈ exp(-H(t)).

**Q: What is Cox Proportional Hazards and what did it show?**  
Cox PH is a semi-parametric regression model: h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...). It estimates which covariates increase or decrease the hazard (no-show risk) without specifying the baseline hazard shape. The concordance index measures how well the model ranks patients by risk — values above 0.5 beat random ranking.

**Q: What practical threshold did you identify?**  
Using the daily no-show rate by lead time, I found a point beyond which the no-show rate exceeds the median. Clinics can use this: patients booked further ahead than this threshold should receive SMS reminders or flexible rescheduling options.

---

## 11. Causal Inference & SMS Budget Optimizer (NB10)

**Q: Why causal inference instead of just prediction?**  
Prediction tells us *who* will likely no-show, but not *why* or *what to do about it*. Causal inference answers: "does sending an SMS reminder actually *cause* patients to attend, or is it just correlated?" This moves the project from predictive to prescriptive analytics.

**Q: What is IPW-ATE and how did you compute it?**  
Inverse Propensity Weighting estimates the Average Treatment Effect by re-weighting outcomes to correct for selection bias. First, I estimated propensity scores (probability of receiving SMS given covariates) using logistic regression. Then the ATE = weighted mean of treated outcomes − weighted mean of control outcomes. Propensity scores were clipped to [0.05, 0.95] to avoid extreme weights. Reference: Imbens & Rubin (2015).

**Q: What is doubly-robust estimation?**  
It combines an outcome model with a propensity score model. Even if one model is misspecified, the ATE estimate is still consistent — you get "two chances" to get it right. This is important because we can never be sure our propensity model is perfect.

**Q: What is CATE and how does it relate to SMS targeting?**  
Conditional Average Treatment Effect (CATE) is the treatment effect for each individual patient. I used a T-Learner approach: train separate models on treated and control groups, then CATE = predicted_outcome_treated − predicted_outcome_control. Patients with high CATE benefit most from SMS reminders.

**Q: How does the budget-constrained optimizer work?**  
Given a limited SMS budget (say 500 messages), the optimizer ranks patients by CATE score (descending) and sends SMS only to the top-ranked patients until the budget runs out. This maximises the total uplift in attendance for the available budget.

**Q: What is a Qini curve?**  
The Qini curve is to uplift modelling what the ROC curve is to classification. It plots cumulative uplift vs. fraction of population targeted. The area between the Qini curve and the random line (AUUC) measures how well the model identifies patients who truly benefit from SMS treatment. Reference: Gutierrez & Gerardy (2017).

---

## 12. Uncertainty-Aware Optimization (NB11)

**Q: How do conformal prediction bounds feed into overbooking decisions?**  
Conformal prediction (from MAPIE) gives a prediction set for each patient — either {show}, {no-show}, or {show, no-show} (uncertain). By counting patients in each set, I compute upper/lower bounds on how many will attend. The overbooking decision then uses these bounds rather than a single-point estimate.

**Q: What is the Newsvendor model and why is it relevant?**  
The Newsvendor is a classic operations research model: how many extra units to stock when demand is uncertain. Here, "demand" is attendance count and "stock" is booked slots. The critical ratio = C_idle / (C_idle + C_overflow) determines the optimal quantile of the attendance distribution to target. With our costs: idle slot = ₹150, overflow = ₹100 (wait + overtime) → critical ratio ≈ 0.60, meaning we should book to the 60th percentile of expected attendance. Reference: Birge & Louveaux, *Introduction to Stochastic Programming*.

**Q: How does this compare to the simulator from NB06?**  
NB06's Monte Carlo simulator evaluates fixed and risk-based strategies by brute force. The Newsvendor gives an analytically optimal solution using the uncertainty bounds. In practice, the Newsvendor and risk-based dynamic strategies perform similarly, but the Newsvendor has a theoretical optimality guarantee under its modelling assumptions.

---

## 13. Deep Tabular Models & Statistical Tests (NB12)

**Q: Why test deep learning on tabular data?**  
Recent literature (Grinsztajn et al., NeurIPS 2022) shows that tree-based models often match or beat deep learning on medium-sized tabular datasets. By testing TabNet and MLP against LightGBM and XGBoost on our data, I can scientifically determine whether the ROC-AUC ceiling (~0.64) is a modelling limitation or a dataset property.

**Q: What is TabNet?**  
TabNet (Arik & Pfister, 2021) uses sequential attention to select features at each decision step, combining the interpretability of feature selection with the power of deep learning. It's specifically designed for tabular data, unlike generic MLPs.

**Q: What is McNemar's test and when do you use it?**  
McNemar's test compares two classifiers on the same test set. It builds a 2×2 contingency table of their disagreements (A gets it right & B gets it wrong, vs. B gets it right & A gets it wrong). The chi-squared statistic tests whether one model makes significantly fewer errors. If p > 0.05, the difference is not statistically significant. Reference: Dietterich (1998).

**Q: What is DeLong's test?**  
DeLong's test (1988) compares the AUC of two ROC curves from the same test set. It accounts for the correlation between the two AUCs (since they share the same test samples). If p > 0.05, the AUC difference is not statistically significant.

**Q: What was the key finding?**  
No deep model significantly outperformed LightGBM on the no-show dataset. This is consistent with the literature and implies that the ROC-AUC ceiling is a property of the dataset — not a modelling limitation. The strongest predictors (patient motivation, weather, transport) simply aren't recorded.

---

## 14. Fairness Mitigation (NB13)

**Q: What problem does fairness mitigation solve?**  
The fairness audit in NB07 detected potential demographic parity differences. But auditing alone doesn't fix the problem. Fairness mitigation actively adjusts the model to reduce disparities — the difference between diagnosing a problem and treating it.

**Q: What is Exponentiated Gradient and how does it work?**  
Exponentiated Gradient (Agarwal et al., 2018) is a reduction-based approach from the fairlearn library. It converts the constrained optimization problem (minimise error subject to fairness constraints) into a sequence of weighted classification problems. At each iteration, it re-weights samples to penalise fairness violations. The result is a randomised classifier that satisfies the fairness constraint within a specified tolerance (ε = 0.01).

**Q: What fairness constraint did you use?**  
Demographic Parity — the prediction rate should be similar across groups (e.g., male vs female). DPD < 0.10 is generally accepted as fair. I also tracked Equalized Odds Difference (TPR and FPR gaps across groups).

**Q: Is there a trade-off between accuracy and fairness?**  
Yes — this is known as the accuracy-fairness trade-off. A small accuracy drop is typically observed when enforcing fairness constraints. In our case, accuracy dropped by roughly 1-2 percentage points while DPD improved substantially. This is consistent with Barocas et al.'s fairness impossibility theorem — you cannot simultaneously satisfy all fairness criteria.

---

## 15. MLOps / Deployment (NB14)

**Q: Why add MLOps to a data science project?**  
A model that only lives in a notebook has zero real-world impact. By wrapping the model in a FastAPI REST API and logging experiments with MLflow, the project demonstrates the full lifecycle: train → evaluate → log → deploy → serve. This is exactly what happens in production settings.

**Q: How does the FastAPI server work?**  
FastAPI provides three endpoints:
- `/health` – health check (is the server running?)
- `/predict` – single patient prediction (accepts JSON with patient features, returns no-show probability)
- `/predict/batch` – batch prediction (multiple patients at once)

Input validation is handled by Pydantic models, which reject invalid data types automatically. The API returns structured JSON with probability, risk label, and model metadata.

**Q: What does MLflow provide?**  
MLflow tracks experiments: each training run logs model parameters, metrics (accuracy, ROC-AUC, F1), and artefacts (model files, confusion matrices). This makes it easy to compare runs, reproduce results, and roll back to a previous model version.

**Q: How does the dashboard use the trained model?**  
The Streamlit dashboard loads the best trained model from `outputs/models/best_model_calibrated.joblib` using `@st.cache_resource` for efficiency. If the model file doesn't exist (e.g., notebooks haven't been run yet), it falls back to a simple rule-based formula. The sidebar shows which mode is active.

---

## 16. Limitations and Future Work

**Limitations:**
- Single-region dataset (Vitória, Brazil) — may not generalise
- Cost parameters are assumed, not real financial data
- No reason for individual no-shows is recorded
- SMOTE creates synthetic minority samples that don't represent real patients
- Causal inference assumes no unobserved confounders (unverifiable)
- Fairness mitigation only addresses gender; intersectional fairness needs more data

**Future improvements:**
- Multi-clinic / multi-region data for better generalisability
- Real cost data from clinic management systems
- Online/incremental learning to adapt to drift
- DQN instead of tabular Q-Learning for larger state spaces
- A/B testing the SMS uplift recommendation in a live clinic
- Causal forest (grf package) for better CATE estimation
- Model monitoring with concept drift detection in production


---

## 17. Advanced Analytics Dashboard (v3)

**Q: What does the ClinicSense dashboard show?**
The dashboard is a 6-tab Streamlit application that brings together all project
components in one interactive interface:

1. **Executive Overview** – KPI cards (total records, no-show rate, SMS sent,
   unique patients), plus appointment status distribution (donut chart),
   SMS effectiveness (grouped bar), age-group breakdown (pie), weekly
   appointment trend (line), and today's simulated risk distribution.

2. **Patient Risk Assessment** – A capacity gauge showing expected shows vs
   slots, probability histogram with threshold lines, and the full appointment
   schedule table with risk badges and suggested actions (phone/SMS/none).

3. **Resource Optimization** – Doctor workload estimation, slot optimization
   recommendations, a capacity utilisation gauge, hourly patient flow chart,
   and overcrowding alerts when any hour exceeds safe thresholds.

4. **Attendance Analytics** – Prior no-show distribution histogram, risk
   breakdown by age group, SMS effect on predicted probability, repeat
   no-show patient identification, and attendance vs no-show by gender.

5. **What-If Simulator** – An interactive slider to test 0–10 extra bookings.
   Shows expected shows, expected no-shows, and P(exceed capacity) using a
   Normal approximation.  A dual-axis chart and results table help pick
   the optimal overbooking level.

6. **AI Clinic Assistant** – A simple rule-based chatbot that answers natural-
   language questions about today's simulated schedule: patient counts, risk
   levels, SMS statistics, overbooking recommendations, peak hours, etc.

**Q: Why did you choose Plotly over Matplotlib for the dashboard?**
Plotly renders interactive charts (hover tooltips, zoom, pan) directly in the
browser.  For a dashboard where the user explores data, interactivity is much
more useful than static Matplotlib images.  I still use Matplotlib in the
notebooks for publication-quality figures.

**Q: How does the AI Assistant work?**
It's a keyword-matching system, not a large language model.  It scans the user's
question for keywords ("high risk", "SMS", "overbook", "capacity") and routes to
pre-written data queries on the session's DataFrame.  This keeps it fast,
deterministic, and doesn't require any API key.

**Q: Why does the What-If simulator use a Normal approximation?**
With n patients each having an independent show probability p, the total shows
roughly follow a Poisson-Binomial distribution.  For n ≥ 20, the Central Limit
Theorem lets us approximate it with N(μ, σ²) where μ = Σ(1 − pᵢ) and
σ² = Σ pᵢ(1 − pᵢ).  This is fast enough for real-time slider interaction.
