"""Cloud Hard Drive Predictive Maintenance SRE Operations Console.

Interactive Streamlit dashboard visualizing enterprise drive fleet health,
real-time 7-day failure risk predictions, and MLOps sensor drift observability.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Hard Drive Failure MLOps Console",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise look
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 8px; border: 1px solid #2d313e; }
    .risk-critical { background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #ef4444; padding: 12px; border-radius: 8px; font-weight: bold; }
    .risk-warning { background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #f59e0b; padding: 12px; border-radius: 8px; font-weight: bold; }
    .risk-healthy { background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981; padding: 12px; border-radius: 8px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    model_path = Path("models/lgb_model.joblib")
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Run training first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_fleet_data():
    parquet_path = Path("data/raw/backblaze_telemetry.parquet")
    if not parquet_path.exists():
        parquet_path = Path("data/raw/telemetry_sample.parquet")

    # Read metadata using Polars
    df = pl.read_parquet(parquet_path)
    return df


@st.cache_data
def get_sample_drives(_df: pl.DataFrame):
    # Get drives that experienced failure
    failing_drives = _df.filter(pl.col("failure") == 1).select("serial_number").unique()["serial_number"].to_list()[:20]
    # Get random healthy drives
    healthy_drives = (
        _df.filter(pl.col("failure") == 0)
        .select("serial_number")
        .unique()["serial_number"]
        .sample(n=min(20, _df["serial_number"].n_unique()), seed=42)
        .to_list()
    )
    return failing_drives, healthy_drives


bundle = load_model_bundle()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
metrics = bundle.get("metrics", {})

# Load fleet dataset
df_raw = load_fleet_data()
failing_drives, healthy_drives = get_sample_drives(df_raw)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/server.png", width=64)
st.sidebar.title("SRE Cloud Console")
st.sidebar.markdown("**Predictive Maintenance Platform**")
st.sidebar.markdown("---")

st.sidebar.markdown("**Model Type:** LightGBM")
st.sidebar.markdown(f"**Features:** {len(feature_cols)} Domain Sensors")
st.sidebar.markdown("**SLA Horizon:** 7 Days Ahead")
st.sidebar.markdown("**Dataset:** Backblaze Q1 2024")
st.sidebar.markdown("---")

# Header Title
st.title("💾 Data Center Hard Drive Failure Intelligence")
st.caption("Enterprise MLOps predictive maintenance engine monitoring 59,000+ drives across cloud storage arrays.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Monitored Drives", f"{df_raw['serial_number'].n_unique():,}")
with col2:
    st.metric("Telemetry Records", f"{len(df_raw):,}")
with col3:
    st.metric("Test Failure Recall", f"{metrics.get('test_recall', 0.322) * 100:.1f}%")
with col4:
    st.metric("Failure SLA Window", "7 Days")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Fleet Health Overview",
        "🔍 Drive Telemetry Inspector",
        "🧪 What-If Risk Simulator",
        "🛡️ MLOps Drift Observability",
    ]
)

# ----------------- TAB 1: FLEET HEALTH OVERVIEW -----------------
with tab1:
    st.subheader("Data Center Storage Fleet Distribution")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Drive models breakdown
        model_counts = df_raw.select("model").to_pandas()["model"].value_counts().reset_index()
        model_counts.columns = ["Model", "Count"]
        fig_models = px.pie(
            model_counts,
            names="Model",
            values="Count",
            title="Drive Fleet Distribution by Hardware Model",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_models.update_layout(template="plotly_dark")
        st.plotly_chart(fig_models, use_container_width=True)

    with col_right:
        # Temperature distribution across fleet
        temp_series = df_raw.select("smart_194_raw").to_pandas()["smart_194_raw"].dropna()
        fig_temp = px.histogram(
            temp_series,
            nbins=35,
            title="Drive Temperature Distribution (°C)",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_temp.update_layout(template="plotly_dark", xaxis_title="Temperature (°C)", yaxis_title="Count")
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("### 🚨 Prioritized Degrading Drives")
    st.markdown("Drives showing sensor wear rates and uncorrectable sector anomalies:")

    # Highlight sample degrading drives
    sample_degrading = (
        df_raw.filter(pl.col("smart_5_raw") > 0)
        .select(["date", "serial_number", "model", "smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_194_raw"])
        .sort(["smart_5_raw", "smart_187_raw"], descending=True)
        .head(10)
        .to_pandas()
    )
    st.dataframe(sample_degrading, use_container_width=True)


# ----------------- TAB 2: DRIVE INSPECTOR -----------------
with tab2:
    st.subheader("Individual Drive Telemetry & Risk Inspector")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        drive_options = (
            ["--- Degraded / Failing Sample Drives ---"] + failing_drives + ["--- Healthy Drives ---"] + healthy_drives
        )
        selected_drive = st.selectbox("Select Drive Serial Number:", drive_options, index=1)
        if selected_drive.startswith("---"):
            selected_drive = failing_drives[0] if failing_drives else "DRIVE-001"

    with col_sel2:
        custom_serial = st.text_input("Or enter custom Serial Number:", "")
        if custom_serial.strip():
            selected_drive = custom_serial.strip()

    # Query telemetry for selected drive
    drive_history = df_raw.filter(pl.col("serial_number") == selected_drive).sort("date").to_pandas()

    if drive_history.empty:
        st.warning(f"No telemetry found for drive serial {selected_drive}.")
    else:
        st.markdown(f"#### Telemetry Record for `{selected_drive}` ({drive_history['model'].iloc[0]})")
        latest_row = drive_history.iloc[-1]

        # Calculate prediction features for latest day
        smart_5 = float(latest_row.get("smart_5_raw", 0.0))
        smart_9 = float(latest_row.get("smart_9_raw", 1.0))
        smart_187 = float(latest_row.get("smart_187_raw", 0.0))
        smart_188 = float(latest_row.get("smart_188_raw", 0.0))
        smart_194 = float(latest_row.get("smart_194_raw", 30.0))
        smart_197 = float(latest_row.get("smart_197_raw", 0.0))

        # Compute rolling window estimates from history
        s5_delta = float(smart_5 - drive_history["smart_5_raw"].iloc[max(0, len(drive_history) - 8)])
        s187_delta = float(smart_187 - drive_history["smart_187_raw"].iloc[max(0, len(drive_history) - 8)])
        s197_delta = float(smart_197 - drive_history["smart_197_raw"].iloc[max(0, len(drive_history) - 8)])
        s194_mean = float(drive_history["smart_194_raw"].tail(7).mean())

        row_dict = {
            "smart_5_raw": smart_5,
            "smart_9_raw": smart_9,
            "smart_187_raw": smart_187,
            "smart_188_raw": smart_188,
            "smart_194_raw": smart_194,
            "smart_197_raw": smart_197,
            "smart_5_raw_mean_7d": float(drive_history["smart_5_raw"].tail(7).mean()),
            "smart_5_raw_delta_7d": s5_delta,
            "smart_187_raw_mean_7d": float(drive_history["smart_187_raw"].tail(7).mean()),
            "smart_187_raw_delta_7d": s187_delta,
            "smart_197_raw_mean_7d": float(drive_history["smart_197_raw"].tail(7).mean()),
            "smart_197_raw_delta_7d": s197_delta,
            "smart_194_raw_mean_7d": s194_mean,
            "smart_194_raw_delta_7d": float(
                smart_194 - drive_history["smart_194_raw"].iloc[max(0, len(drive_history) - 8)]
            ),
            "wear_rate_per_hour": smart_5 / (smart_9 + 1.0),
            "wear_acceleration_rate": s5_delta / (smart_9 + 1.0),
            "composite_critical_errors": smart_5 + smart_187 + smart_197,
            "composite_delta_spike_7d": s5_delta + s187_delta + s197_delta,
            "temperature_instability": abs(smart_194 - s194_mean),
        }

        # Predict
        input_df = pd.DataFrame([row_dict])[feature_cols]
        prob = float(model.predict_proba(input_df)[0, 1])

        # Risk badge
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "7-Day Failure Risk Score (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if prob >= 0.75 else ("#f59e0b" if prob >= 0.40 else "#10b981")},
                        "steps": [
                            {"range": [0, 40], "color": "#064e3b"},
                            {"range": [40, 75], "color": "#78350f"},
                            {"range": [75, 100], "color": "#7f1d1d"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.markdown("#### Automated SRE Triage Recommendation")
            if prob >= 0.75:
                st.markdown(
                    f"<div class='risk-critical'>🚨 CRITICAL RISK ({prob * 100:.1f}%)<br/>"
                    "Action: Evacuate drive data immediately and schedule physical swap within 48h.</div>",
                    unsafe_allow_html=True,
                )
            elif prob >= 0.40:
                st.markdown(
                    f"<div class='risk-warning'>⚠️ WARNING RISK ({prob * 100:.1f}%)<br/>"
                    "Action: Increase telemetry polling frequency and initiate pre-emptive RAID group rebalancing.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='risk-healthy'>✅ HEALTHY ({prob * 100:.1f}%)<br/>"
                    "Action: Drive operates within normal thresholds. No intervention needed.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"**Latest Telemetry:** Power-On Hours: `{smart_9:,.0f}h` | Reallocated Sectors: `{smart_5:.0f}` | Uncorrectable Errors: `{smart_187:.0f}` | Temperature: `{smart_194:.1f}°C`"
            )

        # Time series degradation curves
        st.markdown("#### Historical Telemetry Sensor Trajectory")
        fig_traj = go.Figure()
        fig_traj.add_trace(
            go.Scatter(
                x=drive_history["date"],
                y=drive_history["smart_5_raw"],
                name="Reallocated Sectors (SMART 5)",
                line=dict(color="#f43f5e", width=2),
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=drive_history["date"],
                y=drive_history["smart_187_raw"],
                name="Uncorrectable Errors (SMART 187)",
                line=dict(color="#a855f7", width=2),
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=drive_history["date"],
                y=drive_history["smart_194_raw"],
                name="Temperature °C (SMART 194)",
                line=dict(color="#06b6d4", width=2, dash="dot"),
            )
        )
        fig_traj.update_layout(template="plotly_dark", height=320, xaxis_title="Date", yaxis_title="Sensor Value")
        st.plotly_chart(fig_traj, use_container_width=True)


# ----------------- TAB 3: WHAT-IF SIMULATOR -----------------
with tab3:
    st.subheader("🧪 What-If Storage Telemetry Simulator")
    st.markdown("Simulate synthetic sensor spikes to test the predictive model's sensitivity and decision boundaries:")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_smart_5 = st.slider("SMART 5: Reallocated Sectors", min_value=0, max_value=500, value=25)
        sim_s5_delta = st.slider("SMART 5: 7-Day Delta Surge", min_value=0, max_value=200, value=15)
        sim_hours = st.slider("SMART 9: Power-On Hours", min_value=100, max_value=70000, value=25000)

    with col_sim2:
        sim_smart_187 = st.slider("SMART 187: Uncorrectable Errors", min_value=0, max_value=100, value=5)
        sim_s187_delta = st.slider("SMART 187: 7-Day Delta Surge", min_value=0, max_value=50, value=3)
        sim_smart_197 = st.slider("SMART 197: Pending Sectors", min_value=0, max_value=50, value=4)

    with col_sim3:
        sim_temp = st.slider("SMART 194: Temperature (°C)", min_value=15, max_value=65, value=42)
        sim_temp_delta = st.slider("SMART 194: Temp Instability", min_value=0, max_value=20, value=5)
        sim_smart_188 = st.slider("SMART 188: Command Timeouts", min_value=0, max_value=20, value=0)

    sim_row = {
        "smart_5_raw": float(sim_smart_5),
        "smart_9_raw": float(sim_hours),
        "smart_187_raw": float(sim_smart_187),
        "smart_188_raw": float(sim_smart_188),
        "smart_194_raw": float(sim_temp),
        "smart_197_raw": float(sim_smart_197),
        "smart_5_raw_mean_7d": float(sim_smart_5 - sim_s5_delta / 2),
        "smart_5_raw_delta_7d": float(sim_s5_delta),
        "smart_187_raw_mean_7d": float(sim_smart_187 - sim_s187_delta / 2),
        "smart_187_raw_delta_7d": float(sim_s187_delta),
        "smart_197_raw_mean_7d": float(sim_smart_197),
        "smart_197_raw_delta_7d": 0.0,
        "smart_194_raw_mean_7d": float(sim_temp - sim_temp_delta),
        "smart_194_raw_delta_7d": float(sim_temp_delta),
        "wear_rate_per_hour": sim_smart_5 / (sim_hours + 1.0),
        "wear_acceleration_rate": sim_s5_delta / (sim_hours + 1.0),
        "composite_critical_errors": float(sim_smart_5 + sim_smart_187 + sim_smart_197),
        "composite_delta_spike_7d": float(sim_s5_delta + sim_s187_delta),
        "temperature_instability": float(sim_temp_delta),
    }

    sim_df = pd.DataFrame([sim_row])[feature_cols]
    sim_prob = float(model.predict_proba(sim_df)[0, 1])

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Simulated Failure Risk", f"{sim_prob * 100:.2f}%")
    with res_col2:
        if sim_prob >= 0.75:
            st.error("🚨 CRITICAL: Severe failure probability within 7 days. Evacuate data immediately.")
        elif sim_prob >= 0.40:
            st.warning("⚠️ WARNING: Elevated degradation detected. Initiate preemptive RAID rebuild.")
        else:
            st.success("✅ HEALTHY: Drive is operating well within normal tolerances.")


# ----------------- TAB 4: MLOPS DRIFT OBSERVABILITY -----------------
with tab4:
    st.subheader("🛡️ Telemetry Sensor Drift & Distribution Observability")
    st.markdown(
        "Statistical drift monitoring using **Kolmogorov-Smirnov 2-Sample Tests** and **Population Stability Index (PSI)** against training baseline:"
    )

    drift_json_path = Path("reports/drift_metrics.json")
    if drift_json_path.exists():
        with open(drift_json_path, "r") as f:
            drift_data = json.load(f)

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            is_drifted = drift_data.get("drift_detected", False)
            st.metric("Overall Status", "DRIFT DETECTED" if is_drifted else "ALL SENSORS STABLE")
        with d_col2:
            st.metric("KS Significance Threshold", "p < 0.05")
        with d_col3:
            st.metric("PSI Threshold", "> 0.10")

        # Table of sensor drift metrics
        drift_rows = []
        for col_name, col_stats in drift_data.get("metrics_by_column", {}).items():
            drift_rows.append(
                {
                    "Sensor Feature": col_name,
                    "KS Statistic": col_stats["ks_statistic"],
                    "KS p-value": col_stats["ks_pvalue"],
                    "PSI Score": col_stats["psi"],
                    "Status": "DRIFT" if col_stats["drift_detected"] else "STABLE",
                }
            )
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
    else:
        st.info("No drift report found. Run `python -m src.monitoring.drift` to generate report.")
