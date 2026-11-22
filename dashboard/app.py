"""
ClinicSense – Advanced Analytics Dashboard (v3)
Multi-tab Streamlit dashboard for no-show risk prediction, clinic resource
optimization, patient analytics, what-if simulation and an AI assistant.

Uses the trained calibrated LightGBM model (Notebook 04). Falls back to
a lightweight heuristic when the .joblib file isn't present.

Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, math, joblib, textwrap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")

# path setup so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD,
    SLOTS_PER_DAY, COST_IDLE_SLOT,
    COST_WAITING_MINUTE, COST_OVERTIME_MINUTE, SLOT_DURATION_MIN,
    MODELS_DIR, RAW_DATA_FILE,
)
from src.feature_engineering import MODEL_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicSense – Advanced Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  inspired by the hospital dashboards from reference images
# I wanted a clean blue-gradient header + card styling.  Spent a while
# tweaking the colours until it looked professional enough for the viva.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* top bar gradient */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0D47A1, #1976D2, #42A5F5);
    }
    /* metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #0D47A1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1565C0 !important;
        font-size: 1.6rem !important;
    }
    /* tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    /* risk badges */
    .risk-high   { background:#FFCDD2; padding:4px 12px; border-radius:6px;
                   color:#C62828; font-weight:bold; }
    .risk-medium { background:#FFF9C4; padding:4px 12px; border-radius:6px;
                   color:#E65100; font-weight:bold; }
    .risk-low    { background:#C8E6C9; padding:4px 12px; border-radius:6px;
                   color:#2E7D32; font-weight:bold; }
    /* chatbot styling */
    .chat-msg-bot {
        background: #E3F2FD;
        border-radius: 12px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 85%;
    }
    .chat-msg-user {
        background: #F5F5F5;
        border-radius: 12px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 85%;
        margin-left: auto;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    """Try calibrated LightGBM first, then the tuned one, else None."""
    for fname in ["best_model_calibrated.joblib", "lightgbm_tuned.joblib"]:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try:
                return joblib.load(fpath), True
            except Exception:
                pass
    return None, False

trained_model, using_real_model = load_trained_model()


# ─────────────────────────────────────────────────────────────────────────────
# Load real dataset for the overview analytics tab
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_dataset():
    """Load the original Kaggle CSV for aggregate analytics."""
    if os.path.exists(RAW_DATA_FILE):
        df = pd.read_csv(RAW_DATA_FILE)
        # standardise column names (the CSV has some weird casing)
        df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
        # rename the target
        if "no_show" in df.columns:
            df["no_show"] = df["no_show"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        elif "no show" in df.columns:
            df.rename(columns={"no show": "no_show"}, inplace=True)
            df["no_show"] = df["no_show"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        # parse dates
        for col in ["scheduledday", "appointmentday"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # age groups
        bins = [0, 5, 18, 29, 39, 49, 150]
        labels = ["Under 5", "5-18", "18-29", "30-39", "40-49", "50+"]
        df["age_group"] = pd.cut(
            df["age"].clip(0, 120), bins=bins, labels=labels, right=True
        )
        return df
    return None

raw_df = load_raw_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 ClinicSense")
st.sidebar.caption("Advanced Analytics Dashboard")
st.sidebar.divider()

n_slots_input = st.sidebar.slider(
    "Base Slots per Day", min_value=10, max_value=30, value=SLOTS_PER_DAY
)
n_patients = st.sidebar.slider(
    "Appointments to Simulate", min_value=5, max_value=50, value=25
)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
run_sim = st.sidebar.button("🔄  Generate New Day", type="primary")

if using_real_model:
    st.sidebar.success("✅ ML model loaded")
else:
    st.sidebar.warning("⚠ Rule-based fallback")

st.sidebar.divider()
st.sidebar.markdown(
    "**Student:** Kadam Akanksha Dinkar  \n"
    "B.Sc. Data Science Honours"
)


# ─────────────────────────────────────────────────────────────────────────────
# Generate a simulated clinic day
# ─────────────────────────────────────────────────────────────────────────────
def _rule_based_prob(row, noise):
    """Fallback heuristic when no model .joblib exists."""
    base = 0.20
    p = (base
         + 0.003 * row["lead_time_days"]
         + 0.06  * row["prior_no_shows"]
         - 0.04  * row["sms_received"]
         - 0.001 * row["age"]
         + noise)
    return np.clip(p, 0.02, 0.95)


@st.cache_data(ttl=0)
def generate_day(n_pat: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ages       = rng.integers(5, 90, n_pat)
    genders    = rng.integers(0, 2, n_pat).astype("int8")
    lead_times = rng.integers(0, 60, n_pat)
    prior_ns   = rng.integers(0, 5, n_pat)
    sms_recv   = rng.integers(0, 2, n_pat).astype("int8")
    hyp        = rng.integers(0, 2, n_pat).astype("int8")
    diab       = rng.integers(0, 2, n_pat).astype("int8")
    alc        = rng.integers(0, 2, n_pat).astype("int8")
    handi      = rng.integers(0, 2, n_pat).astype("int8")
    schol      = rng.integers(0, 2, n_pat).astype("int8")

    age_enc    = np.clip(np.digitize(ages, [12, 17, 35, 60]), 0, 4).astype("int8")
    cond_count = (hyp + diab + alc + handi).astype("int8")
    same_day   = (lead_times == 0).astype("int8")
    appt_wd    = rng.integers(0, 5, n_pat).astype("int8")
    appt_mo    = rng.integers(1, 13, n_pat).astype("int8")
    sched_hr   = rng.integers(6, 20, n_pat).astype("int8")
    is_mon     = (appt_wd == 0).astype("int8")
    is_fri     = (appt_wd == 4).astype("int8")

    prior_appt = rng.integers(0, 15, n_pat)
    # fixed divide-by-zero:  compute only where prior_appt > 0
    prior_rate = np.zeros(n_pat, dtype="float32")
    mask = prior_appt > 0
    if mask.any():
        prior_rate[mask] = (prior_ns[mask] / prior_appt[mask]).astype("float32")
    nbr_freq = rng.uniform(0.005, 0.04, n_pat).astype("float32")

    feat_df = pd.DataFrame({
        "age": ages, "age_group_enc": age_enc, "gender": genders,
        "scholarship": schol, "hypertension": hyp, "diabetes": diab,
        "alcoholism": alc, "handicap": handi, "sms_received": sms_recv,
        "lead_time_days": lead_times, "appt_weekday": appt_wd,
        "appt_month": appt_mo, "sched_hour": sched_hr,
        "is_monday": is_mon, "is_friday": is_fri,
        "prior_appointments": prior_appt, "prior_no_shows": prior_ns,
        "prior_no_show_rate": prior_rate, "condition_count": cond_count,
        "same_day_booking": same_day, "neighbourhood_freq": nbr_freq,
    })
    for col in MODEL_FEATURES:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[MODEL_FEATURES]

    # predictions
    if using_real_model and trained_model is not None:
        try:
            probs = trained_model.predict_proba(feat_df)[:, 1]
        except Exception:
            noise = rng.uniform(-0.05, 0.05, n_pat)
            probs = np.array([_rule_based_prob(feat_df.iloc[i], noise[i])
                              for i in range(n_pat)])
    else:
        noise = rng.uniform(-0.05, 0.05, n_pat)
        probs = np.array([_rule_based_prob(feat_df.iloc[i], noise[i])
                          for i in range(n_pat)])

    # start times spread across the day 8am-5pm
    start_minutes = np.sort(rng.integers(480, 1020, n_pat))   # 8:00 to 17:00
    times = [f"{m // 60:02d}:{m % 60:02d}" for m in start_minutes]

    df = pd.DataFrame({
        "PatientID":     [f"P{1000+i}" for i in range(n_pat)],
        "Name":          [f"Patient {chr(65 + i % 26)}{i+1}" for i in range(n_pat)],
        "Age":           ages,
        "Gender":        ["M" if g else "F" for g in genders],
        "LeadTime":      lead_times,
        "PriorNoShows":  prior_ns,
        "SMSReceived":   sms_recv,
        "Hypertension":  hyp,
        "Diabetes":      diab,
        "SchedHour":     sched_hr,
        "ApptTime":      times,
        "NoShowProb":    np.clip(probs, 0.02, 0.95),
        "PriorAppt":     prior_appt,
    })
    df["AttendProb"] = 1 - df["NoShowProb"]
    df["Risk"] = df["NoShowProb"].apply(
        lambda p: "High"   if p >= RISK_HIGH_THRESHOLD
             else "Medium" if p >= RISK_LOW_THRESHOLD
             else "Low"
    )
    # age group for analytics
    bins   = [0, 5, 18, 29, 39, 49, 150]
    labels = ["Under 5", "5-18", "18-29", "30-39", "40-49", "50+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df


# generate or reuse
if run_sim or "day_df" not in st.session_state:
    st.session_state["day_df"]  = generate_day(n_patients, random_seed)
    st.session_state["n_slots"] = n_slots_input

df      = st.session_state["day_df"]
n_slots = st.session_state["n_slots"]


# ─────────────────────────────────────────────────────────────────────────────
# Precompute common aggregates
# ─────────────────────────────────────────────────────────────────────────────
expected_shows   = df["AttendProb"].sum()
expected_noshows = df["NoShowProb"].sum()
rec_extra        = max(0, round(expected_noshows))
total_booked     = len(df)
high_risk_cnt    = (df["Risk"] == "High").sum()
med_risk_cnt     = (df["Risk"] == "Medium").sum()
low_risk_cnt     = (df["Risk"] == "Low").sum()
utilisation      = min(expected_shows, n_slots) / n_slots * 100
no_show_rate_sim = expected_noshows / total_booked * 100 if total_booked else 0
sms_sent_cnt     = int(df["SMSReceived"].sum())


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
st.title("🏥 ClinicSense – Advanced Analytics Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview",
    "🎯 Patient Risk Assessment",
    "⚙️ Resource Optimization",
    "📈 Attendance Analytics",
    "🔬 What-If Simulator",
    "🤖 AI Clinic Assistant",
])


# ===================== TAB 1 – Executive Overview ============================
with tab1:
    st.markdown("### Executive Overview")

    # -- KPI Row (inspired by the Hospital Management Analysis Dashboard) --
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    if raw_df is not None:
        total_records = len(raw_df)
        overall_ns_rate = raw_df["no_show"].mean() * 100 if "no_show" in raw_df else 20.0
        total_patients  = raw_df["patientid"].nunique() if "patientid" in raw_df else total_records
        sms_total       = int(raw_df["sms_received"].sum()) if "sms_received" in raw_df else 0
    else:
        total_records, overall_ns_rate, total_patients, sms_total = 110527, 20.2, 62299, 35482

    k1.metric("Total Records", f"{total_records:,}")
    k2.metric("No-Show Rate", f"{overall_ns_rate:.1f}%")
    k3.metric("SMS Sent", f"{sms_total:,}")
    k4.metric("Unique Patients", f"{total_patients:,}")
    k5.metric("Today Booked", f"{total_booked}")
    k6.metric("Est. Utilisation", f"{utilisation:.0f}%")

    st.divider()

    # -- Row 2: three charts side by side --
    c1, c2, c3 = st.columns(3)

    with c1:
        # Appointment Status donut  (like the reference image)
        if raw_df is not None and "no_show" in raw_df.columns:
            status_counts = raw_df["no_show"].value_counts().rename(
                {0: "Showed-up", 1: "No-show"}
            )
        else:
            status_counts = pd.Series({"Showed-up": 88208, "No-show": 22319})
        fig_donut = px.pie(
            names=status_counts.index, values=status_counts.values,
            hole=0.45, title="Appointment Status Distribution",
            color_discrete_sequence=["#1976D2", "#EF5350"],
        )
        fig_donut.update_traces(textinfo="label+percent", textposition="outside")
        fig_donut.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10),
                                showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Appointment Status by SMS  (stacked bar)
        if raw_df is not None and "sms_received" in raw_df.columns:
            sms_grp = (raw_df.groupby(["sms_received", "no_show"])
                       .size().reset_index(name="count"))
            sms_grp["sms_label"] = sms_grp["sms_received"].map({0: "No SMS", 1: "SMS Sent"})
            sms_grp["status"]    = sms_grp["no_show"].map({0: "Showed-up", 1: "No-show"})
        else:
            sms_grp = pd.DataFrame({
                "sms_label": ["No SMS","No SMS","SMS Sent","SMS Sent"],
                "status":    ["Showed-up","No-show","Showed-up","No-show"],
                "count":     [75045, 20680, 13163, 1639],
            })
        fig_sms = px.bar(
            sms_grp, x="sms_label", y="count", color="status",
            barmode="group", title="Appointment Status by SMS",
            color_discrete_map={"Showed-up": "#1976D2", "No-show": "#EF5350"},
        )
        fig_sms.update_layout(height=340, margin=dict(t=40, b=10),
                              xaxis_title="", yaxis_title="Count",
                              legend_title="")
        st.plotly_chart(fig_sms, use_container_width=True)

    with c3:
        # No-Show Rate by Age Group  (donut)
        if raw_df is not None and "age_group" in raw_df.columns:
            age_counts = raw_df["age_group"].value_counts().sort_index()
        else:
            age_counts = pd.Series({
                "Under 5": 10000, "5-18": 14000, "18-29": 37000,
                "30-39": 17000, "40-49": 15000, "50+": 17000,
            })
        fig_age = px.pie(
            names=age_counts.index, values=age_counts.values,
            hole=0.4, title="No-Show Rate by Age Group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_age.update_traces(textinfo="label+percent", textposition="outside")
        fig_age.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10),
                              showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    # -- Row 3: Monthly trend + Risk distribution --
    c4, c5 = st.columns([3, 2])

    with c4:
        # Monthly Appointment Trend (line chart – like in the reference)
        if raw_df is not None and "appointmentday" in raw_df.columns:
            monthly = (raw_df.set_index("appointmentday")
                       .resample("W")["no_show"].count().reset_index())
            monthly.columns = ["Week", "Appointments"]
        else:
            monthly = pd.DataFrame({
                "Week": pd.date_range("2016-04-29", periods=6, freq="W"),
                "Appointments": [4200, 3900, 4500, 4300, 4100, 4400],
            })
        fig_trend = px.line(
            monthly, x="Week", y="Appointments",
            title="Appointment Volume Over Time",
            markers=True,
        )
        fig_trend.update_traces(line_color="#1976D2", line_width=2.5)
        fig_trend.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)

    with c5:
        # Today's simulated risk breakdown
        risk_data = pd.DataFrame({
            "Risk": ["Low", "Medium", "High"],
            "Count": [low_risk_cnt, med_risk_cnt, high_risk_cnt],
        })
        fig_risk = px.bar(
            risk_data, x="Risk", y="Count",
            color="Risk",
            color_discrete_map={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"},
            title="Today's Risk Distribution",
        )
        fig_risk.update_layout(height=320, margin=dict(t=40, b=10),
                               showlegend=False, xaxis_title="", yaxis_title="Patients")
        for i, row in risk_data.iterrows():
            fig_risk.add_annotation(x=row["Risk"], y=row["Count"],
                                    text=str(row["Count"]),
                                    showarrow=False, yshift=12, font_size=14,
                                    font_color="black")
        st.plotly_chart(fig_risk, use_container_width=True)


# ===================== TAB 2 – Patient Risk Assessment =======================
with tab2:
    st.markdown("### 🎯 Patient Risk Assessment")

    # -- Overbooking recommendation --
    ob1, ob2 = st.columns([1, 2])
    with ob1:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Base capacity** | {n_slots} slots |
        | **Booked today** | {total_booked} |
        | **Expected no-shows** | {expected_noshows:.1f} |
        | **Recommended extras** | **{rec_extra}** |
        """)
        if rec_extra == 0:
            st.success("No extra bookings needed today.")
        elif rec_extra <= 2:
            st.info(f"Accept **{rec_extra}** extra patient(s).")
        else:
            st.warning(f"Accept **{rec_extra}** extra(s) — monitor wait times!")

    with ob2:
        # capacity gauge (plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(expected_shows, 1),
            delta={"reference": n_slots, "relative": False},
            title={"text": "Expected Shows vs Capacity"},
            gauge={
                "axis": {"range": [0, n_slots + rec_extra + 5]},
                "bar": {"color": "#1976D2"},
                "steps": [
                    {"range": [0, n_slots * 0.7], "color": "#E8F5E9"},
                    {"range": [n_slots * 0.7, n_slots], "color": "#FFF9C4"},
                    {"range": [n_slots, n_slots + rec_extra + 5], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.8,
                    "value": n_slots,
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # -- Probability distribution --
    fig_hist = px.histogram(
        df, x="NoShowProb", nbins=20,
        title="No-Show Probability Distribution",
        color_discrete_sequence=["#1976D2"],
    )
    fig_hist.add_vline(x=RISK_LOW_THRESHOLD, line_dash="dash", line_color="orange",
                       annotation_text=f"Medium ({RISK_LOW_THRESHOLD})")
    fig_hist.add_vline(x=RISK_HIGH_THRESHOLD, line_dash="dash", line_color="red",
                       annotation_text=f"High ({RISK_HIGH_THRESHOLD})")
    fig_hist.update_layout(height=300, xaxis_title="P(No-Show)", yaxis_title="Patients")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # -- Patient schedule table --
    st.subheader("🗓️ Today's Appointment Schedule")

    def risk_badge(r):
        return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(r, r)

    def suggest_action(row):
        if row["Risk"] == "High":    return "📞 Phone + SMS"
        elif row["Risk"] == "Medium": return "📱 SMS reminder"
        return "✓ No action"

    disp = df[["PatientID","Name","Age","Gender","ApptTime",
               "LeadTime","SMSReceived","NoShowProb","Risk"]].copy()
    disp["NoShowProb"] = disp["NoShowProb"].round(3)
    disp["Risk"]       = disp["Risk"].apply(risk_badge)
    disp["Action"]     = df.apply(suggest_action, axis=1)
    disp = disp.rename(columns={
        "PatientID": "ID", "LeadTime": "Lead(d)",
        "SMSReceived": "SMS", "NoShowProb": "P(No-Show)"
    })
    st.dataframe(disp.set_index("ID"), use_container_width=True, height=400)

    st.download_button(
        label="⬇️ Download Schedule CSV",
        data=disp.to_csv(index=True),
        file_name="todays_schedule.csv", mime="text/csv",
    )


# ===================== TAB 3 – Resource Optimization =========================
with tab3:
    st.markdown("### ⚙️ Intelligent Clinic Resource Optimization")
    st.caption("Estimates based on today's simulated appointment list")

    # -- Doctor workload estimation --
    # I'm assuming 3 doctors share the slot pool.  Each doctor handles
    # roughly n_slots/3 appointments.  Textbook ref: Cayirli & Veral (2003)
    n_doctors = st.slider("Number of doctors on duty", 1, 6, 3, key="ndocs")
    slots_per_doc = n_slots / n_doctors
    exp_per_doc   = expected_shows / n_doctors

    st.divider()

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("#### 👨‍⚕️ Doctor Workload")
        doc_data = pd.DataFrame({
            "Doctor": [f"Dr. {chr(65+i)}" for i in range(n_doctors)],
            "Assigned Slots": [int(slots_per_doc)] * n_doctors,
            "Expected Patients": [round(exp_per_doc, 1)] * n_doctors,
            "Load %": [round(exp_per_doc / slots_per_doc * 100, 1)] * n_doctors,
        })
        st.dataframe(doc_data, hide_index=True, use_container_width=True)

    with d2:
        st.markdown("#### 📅 Slot Optimization")
        optimal_slots = math.ceil(expected_shows * 1.05)   # 5% buffer
        st.metric("Optimal Slots Today", optimal_slots)
        st.metric("Current Slots", n_slots)
        diff = optimal_slots - n_slots
        if diff > 0:
            st.warning(f"Consider opening **{diff}** more slot(s).")
        elif diff < 0:
            st.info(f"Could free up **{abs(diff)}** slot(s).")
        else:
            st.success("Slot count matches demand perfectly.")

    with d3:
        st.markdown("#### 🏥 Capacity Utilisation")
        fig_util = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(utilisation, 1),
            number={"suffix": "%"},
            title={"text": "Utilisation"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1976D2"},
                "steps": [
                    {"range": [0, 50],  "color": "#FFCDD2"},
                    {"range": [50, 85], "color": "#FFF9C4"},
                    {"range": [85, 100],"color": "#C8E6C9"},
                ],
            },
        ))
        fig_util.update_layout(height=250, margin=dict(t=50, b=10))
        st.plotly_chart(fig_util, use_container_width=True)

    st.divider()

    # -- Expected patient flow by hour --
    st.markdown("#### ⏰ Expected Patient Flow by Hour")
    hour_df = df.copy()
    hour_df["Hour"] = hour_df["SchedHour"]
    flow = hour_df.groupby("Hour").agg(
        Booked=("PatientID", "count"),
        ExpectedShows=("AttendProb", "sum"),
    ).reset_index()
    flow["ExpectedShows"] = flow["ExpectedShows"].round(1)

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=flow["Hour"], y=flow["Booked"],
        name="Booked", marker_color="#BBDEFB",
    ))
    fig_flow.add_trace(go.Scatter(
        x=flow["Hour"], y=flow["ExpectedShows"],
        name="Expected Shows", mode="lines+markers",
        line=dict(color="#1565C0", width=3),
    ))
    fig_flow.update_layout(
        height=320, xaxis_title="Hour of Day", yaxis_title="Patients",
        barmode="overlay", legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # overcrowding alert
    peak_hour = flow.loc[flow["Booked"].idxmax()]
    if peak_hour["Booked"] > slots_per_doc * 1.3:
        st.error(
            f"⚠️ **Overcrowding Alert:** Hour {int(peak_hour['Hour'])}:00 has "
            f"{int(peak_hour['Booked'])} bookings — above safe threshold!"
        )
    else:
        st.success("No overcrowding detected for today's schedule.")


# ===================== TAB 4 – Attendance Analytics ==========================
with tab4:
    st.markdown("### 📈 Patient Attendance History Analytics")

    a1, a2 = st.columns(2)

    with a1:
        # Prior no-show count distribution (from simulated data)
        fig_prior = px.histogram(
            df, x="PriorNoShows", nbins=6,
            title="Distribution of Prior No-Shows",
            color_discrete_sequence=["#5C6BC0"],
        )
        fig_prior.update_layout(height=320, xaxis_title="Prior No-Shows",
                                yaxis_title="Patients")
        st.plotly_chart(fig_prior, use_container_width=True)

    with a2:
        # Risk breakdown by age group
        age_risk = df.groupby(["AgeGroup", "Risk"]).size().reset_index(name="Count")
        fig_ar = px.bar(
            age_risk, x="AgeGroup", y="Count", color="Risk",
            barmode="stack", title="Risk Level by Age Group",
            color_discrete_map={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"},
        )
        fig_ar.update_layout(height=320, xaxis_title="Age Group", yaxis_title="Patients")
        st.plotly_chart(fig_ar, use_container_width=True)

    st.divider()

    b1, b2 = st.columns(2)

    with b1:
        # SMS effectiveness
        sms_eff = df.groupby("SMSReceived").agg(
            avg_prob=("NoShowProb", "mean"),
            count=("PatientID", "count"),
        ).reset_index()
        sms_eff["SMSReceived"] = sms_eff["SMSReceived"].map({0: "No SMS", 1: "SMS Sent"})
        fig_sms_eff = px.bar(
            sms_eff, x="SMSReceived", y="avg_prob",
            title="SMS Effect on No-Show Probability",
            text=sms_eff["avg_prob"].round(3),
            color_discrete_sequence=["#26A69A"],
        )
        fig_sms_eff.update_layout(height=300, xaxis_title="",
                                  yaxis_title="Avg P(No-Show)")
        st.plotly_chart(fig_sms_eff, use_container_width=True)

    with b2:
        # Repeat no-show patterns  (patients with 2+ prior no-shows)
        repeat = df[df["PriorNoShows"] >= 2]
        st.markdown("#### 🔁 Repeat No-Show Patients")
        st.metric("Patients with 2+ Prior No-Shows", len(repeat))
        if len(repeat) > 0:
            st.metric("Their Avg P(No-Show)", f"{repeat['NoShowProb'].mean():.3f}")
            st.caption(
                "These patients might benefit from personalised interventions "
                "(phone calls, flexible rescheduling, transportation assistance)."
            )
        else:
            st.info("No repeat offenders in today's list!")

    st.divider()

    # Attendance rate by gender
    if raw_df is not None and "gender" in raw_df.columns and "no_show" in raw_df.columns:
        gen_rate = raw_df.groupby("gender")["no_show"].mean().reset_index()
        gen_rate.columns = ["Gender", "No-Show Rate"]
        gen_rate["Attendance Rate"] = 1 - gen_rate["No-Show Rate"]
        fig_gen = px.bar(
            gen_rate, x="Gender", y=["Attendance Rate", "No-Show Rate"],
            barmode="stack", title="Attendance vs No-Show by Gender",
            color_discrete_sequence=["#1976D2", "#EF5350"],
        )
        fig_gen.update_layout(height=300, yaxis_title="Proportion", xaxis_title="")
        st.plotly_chart(fig_gen, use_container_width=True)


# ===================== TAB 5 – What-If Simulator =============================
with tab5:
    st.markdown("### 🔬 Smart Appointment & Resource Optimization")
    st.caption(
        "Adjust the parameters below to see how overbooking affects "
        "expected outcomes.  The simulator uses today's patient list."
    )

    sim1, sim2 = st.columns([1, 2])

    with sim1:
        extra_bookings = st.slider(
            "Extra bookings above capacity", 0, 10, rec_extra, key="wb_extra"
        )
        capacity = n_slots
        total_w  = total_booked + extra_bookings

        # expected attendance at the new total
        # simply scale: if we add extra patients they'd have ~avg no-show prob
        avg_ns_prob   = df["NoShowProb"].mean()
        new_exp_shows = expected_shows + extra_bookings * (1 - avg_ns_prob)
        new_exp_ns    = expected_noshows + extra_bookings * avg_ns_prob

        # probability of exceeding capacity (binomial approx)
        # P(shows > capacity) ≈ using normal approx
        mu    = new_exp_shows
        sigma = math.sqrt(total_w * avg_ns_prob * (1 - avg_ns_prob)) if total_w > 0 else 1
        from scipy.stats import norm
        p_exceed = 1 - norm.cdf(capacity, loc=mu, scale=max(sigma, 0.01))

        st.metric("Total Bookings", total_w)
        st.metric("Expected Shows", f"{new_exp_shows:.1f}")
        st.metric("Expected No-Shows", f"{new_exp_ns:.1f}")
        st.metric("P(Exceed Capacity)", f"{p_exceed:.1%}")

        if p_exceed > 0.3:
            st.error("⚠️ High risk of overcrowding!")
        elif p_exceed > 0.1:
            st.warning("Moderate overflow risk.")
        else:
            st.success("Overflow risk is acceptable.")

    with sim2:
        # Simulation across different extra-booking levels
        extras_range = list(range(0, 11))
        sim_results = []
        for ex in extras_range:
            tw = total_booked + ex
            es = expected_shows + ex * (1 - avg_ns_prob)
            sig = math.sqrt(tw * avg_ns_prob * (1 - avg_ns_prob)) if tw > 0 else 1
            pe = 1 - norm.cdf(capacity, loc=es, scale=max(sig, 0.01))
            idle = max(0, capacity - es)
            sim_results.append({
                "Extra Bookings": ex,
                "Expected Shows": round(es, 1),
                "P(Exceed)": round(pe, 3),
                "Expected Idle Slots": round(idle, 1),
                "Revenue Loss (Idle)": round(idle * COST_IDLE_SLOT, 0),
            })
        sim_df = pd.DataFrame(sim_results)

        # dual-axis chart
        fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sim.add_trace(
            go.Bar(x=sim_df["Extra Bookings"], y=sim_df["Expected Shows"],
                   name="Expected Shows", marker_color="#1976D2", opacity=0.7),
            secondary_y=False,
        )
        fig_sim.add_trace(
            go.Scatter(x=sim_df["Extra Bookings"], y=sim_df["P(Exceed)"],
                       name="P(Exceed Capacity)", mode="lines+markers",
                       line=dict(color="#F44336", width=3)),
            secondary_y=True,
        )
        fig_sim.add_hline(y=capacity, line_dash="dash", line_color="green",
                          annotation_text=f"Capacity ({capacity})")
        fig_sim.update_layout(
            title="Overbooking Simulation",
            height=380,
            legend=dict(orientation="h", y=1.15),
        )
        fig_sim.update_yaxes(title_text="Expected Shows", secondary_y=False)
        fig_sim.update_yaxes(title_text="P(Exceed)", secondary_y=True)
        fig_sim.update_xaxes(title_text="Extra Bookings")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    # show the simulation table
    st.markdown("#### Simulation Results Table")
    st.dataframe(sim_df, hide_index=True, use_container_width=True)

    # recommendation box
    # pick the row where P(Exceed) is closest to 0.15 (acceptable threshold)
    best_idx = (sim_df["P(Exceed)"] - 0.15).abs().idxmin()
    best_row = sim_df.iloc[best_idx]
    st.info(
        f"**Recommendation:** Accept **{int(best_row['Extra Bookings'])}** "
        f"extra booking(s) for a {best_row['P(Exceed)']:.1%} overflow probability "
        f"and ~{best_row['Expected Idle Slots']:.0f} expected idle slots."
    )


# ===================== TAB 6 – AI Clinic Assistant ===========================
with tab6:
    st.markdown("### 🤖 AI Clinic Assistant")
    st.caption(
        "Ask questions about today's schedule and the assistant will "
        "look up the data for you.  Try: *How many high-risk patients?*"
    )

    # init chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant",
             "text": "Hello! I'm your ClinicSense AI assistant. "
                     "Ask me anything about today's appointments! 🏥"}
        ]

    def answer_question(q: str) -> str:
        """Simple rule-based Q&A.  Matches keywords to data queries."""
        q_low = q.lower().strip()

        # -- counts --
        if any(w in q_low for w in ["how many patient", "total patient",
                                     "how many appointment", "total appointment"]):
            return (f"Today we have **{total_booked}** appointments booked "
                    f"across {n_slots} available slots.")

        if "high risk" in q_low or "high-risk" in q_low:
            names = df.loc[df["Risk"]=="High", "Name"].tolist()
            if names:
                return (f"There are **{high_risk_cnt}** high-risk patients: "
                        f"{', '.join(names[:5])}{'...' if len(names)>5 else ''}. "
                        f"Consider phone reminders for them.")
            return "No high-risk patients today! 🎉"

        if "medium risk" in q_low:
            return f"There are **{med_risk_cnt}** medium-risk patients today."

        if "low risk" in q_low:
            return f"There are **{low_risk_cnt}** low-risk patients today."

        # -- no-show rate --
        if "no-show rate" in q_low or "no show rate" in q_low or "noshow" in q_low:
            return f"Today's estimated no-show rate is **{no_show_rate_sim:.1f}%**."

        # -- SMS --
        if "sms" in q_low:
            return (f"**{sms_sent_cnt}** out of {total_booked} patients have "
                    f"received an SMS reminder.")

        # -- overbooking --
        if "overbook" in q_low or "extra booking" in q_low or "recommendation" in q_low:
            return (f"Based on today's risk profile, I recommend accepting "
                    f"**{rec_extra}** extra booking(s). Expected no-shows: "
                    f"{expected_noshows:.1f}.")

        # -- utilisation --
        if "utilis" in q_low or "utiliz" in q_low or "capacity" in q_low:
            return f"Expected utilisation today is **{utilisation:.1f}%** of {n_slots} slots."

        # -- peak hour --
        if "peak" in q_low or "busiest" in q_low or "busy" in q_low:
            hr_counts = df["SchedHour"].value_counts()
            peak = hr_counts.idxmax()
            return f"The busiest hour is **{peak}:00** with {hr_counts.max()} appointments."

        # -- oldest / youngest --
        if "oldest" in q_low:
            oldest = df.loc[df["Age"].idxmax()]
            return f"The oldest patient is **{oldest['Name']}**, age {oldest['Age']}."
        if "youngest" in q_low:
            youngest = df.loc[df["Age"].idxmin()]
            return f"The youngest patient is **{youngest['Name']}**, age {youngest['Age']}."

        # -- specific patient lookup --
        if q_low.startswith("p1") or q_low.startswith("patient"):
            # try to find by ID
            for _, row in df.iterrows():
                if row["PatientID"].lower() in q_low or row["Name"].lower() in q_low:
                    return (f"**{row['Name']}** ({row['PatientID']}): "
                            f"Age {row['Age']}, Risk {row['Risk']}, "
                            f"P(No-Show) = {row['NoShowProb']:.3f}")

        # -- model info --
        if "model" in q_low or "algorithm" in q_low:
            if using_real_model:
                return ("The dashboard uses a **calibrated LightGBM** model "
                        "trained on the Kaggle Medical Appointment dataset "
                        "(Notebooks 03–04).")
            return "Currently using a rule-based fallback. Train the model to get ML predictions."

        # -- help / default --
        if "help" in q_low or "what can" in q_low:
            return textwrap.dedent("""
            I can answer questions like:
            - *How many patients today?*
            - *How many high-risk patients?*
            - *What is today's no-show rate?*
            - *How many SMS sent?*
            - *What is the overbooking recommendation?*
            - *What is the utilisation?*
            - *Which hour is busiest?*
            - *Who is the oldest patient?*
            - *What model are we using?*
            """)

        return ("I'm not sure about that. Try asking about **patient counts**, "
                "**risk levels**, **SMS**, **overbooking**, or **utilisation**. "
                "Type *help* for all available questions.")

    # render chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["text"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["text"])

    # input box
    user_input = st.chat_input("Ask me anything about today's schedule...")
    if user_input:
        st.session_state["chat_history"].append(
            {"role": "user", "text": user_input}
        )
        response = answer_question(user_input)
        st.session_state["chat_history"].append(
            {"role": "assistant", "text": response}
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
model_note = ("Probabilities from the **calibrated LightGBM model** (NB 03–04)."
              if using_real_model
              else "Rule-based estimates. Run Notebooks 03–04 to train the model.")
st.markdown(f"""
**ClinicSense** | B.Sc. Data Science Honours | Kadam Akanksha Dinkar  
*{model_note}*  
Dashboard v3 – Advanced Analytics Edition
""")
