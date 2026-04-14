import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
from core.predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoHazard ML", layout="wide", page_icon="🌍")
st.title("🌍 Geo Hazard Prediction System")
st.markdown("Analyze geological hazard risk using ML models (Random Forest & XGBoost).")

# ── Historical heatmap ────────────────────────────────────────────────────────
st.subheader("🌍 Historical Hazard Heatmap")

df_grid = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/grid_dataset.csv"))

fig_hist = px.density_mapbox(
    df_grid, lat="lat_grid", lon="lon_grid", z="inform_risk_score",
    radius=6, center=dict(lat=20, lon=0), zoom=1, height=500,
    color_continuous_scale="YlOrRd"
)
fig_hist.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_hist, use_container_width=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("⚙️ Enter Parameters for Prediction")

col1, col2 = st.columns(2)

with col1:
    eq_count            = st.number_input("Earthquake Count",      min_value=0,   value=1,    step=1)
    eq_mag_mean         = st.number_input("Avg Magnitude",          value=4.5,     step=0.1)
    eq_depth_mean       = st.number_input("Avg Depth (km)",         value=10.0,    step=1.0)
    has_seismic_activity= st.selectbox("Seismic Activity",         [1, 0],
                                        format_func=lambda x: "Yes" if x else "No")
    eq_frequency_score  = st.number_input("Frequency Score",        value=1.0,     step=0.1)
    continent           = st.selectbox("Continent",
                                        ["Africa","Asia","Europe",
                                         "North America","Oceania","South America"])

with col2:
    seismic_energy_proxy= st.number_input("Seismic Energy Proxy",   value=1000.0,  step=100.0)
    mag_depth_ratio     = st.number_input("Magnitude / Depth Ratio",value=0.2,     step=0.01)
    depth_category      = st.selectbox("Depth Category",
                                        [0, 1, 2, 3],
                                        format_func=lambda x: {
                                            0: "0 — No data",
                                            1: "1 — Shallow (< 70 km)",
                                            2: "2 — Intermediate (70–300 km)",
                                            3: "3 — Deep (> 300 km)"
                                        }[x])
    lat_abs             = st.number_input("Absolute Latitude (0–90)",value=10.0, min_value=0.0, max_value=90.0, step=0.5)
    lon                 = st.number_input("Longitude (-180 to 180)", value=78.0, min_value=-180.0, max_value=180.0, step=0.5)
    neighbor_eq_count   = st.number_input("Neighbour EQ Count",      value=1.0,     step=0.1)

model_choice = st.radio("Model", ["XGBoost (recommended)", "Random Forest"])
model_type   = "xgb" if "XGBoost" in model_choice else "rf"

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = {
        "eq_count":              eq_count,
        "eq_mag_mean":           eq_mag_mean,
        "eq_depth_mean":         eq_depth_mean,
        "has_seismic_activity":  has_seismic_activity,
        "eq_frequency_score":    eq_frequency_score,
        "seismic_energy_proxy":  seismic_energy_proxy,
        "mag_depth_ratio":       mag_depth_ratio,
        "depth_category":        depth_category,
        "lat_abs":               lat_abs,
        "neighbor_eq_count":     neighbor_eq_count,
        "continent":             continent,   # string — predict.py handles encoding
    }

    result = predict(input_data, model=model_type)

    # ── Result display ────────────────────────────────────────────────────────
    st.subheader("📊 Prediction Result")

    RISK_COLOR = {
        "Very Low":  "🟢", "Low": "🟡",
        "Medium":    "🟠", "High": "🔴", "Very High": "🚨"
    }
    icon = RISK_COLOR.get(result["prediction"], "⚪")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level",  f"{icon} {result['prediction']}")
    c2.metric("Confidence",  f"{result['confidence']*100:.1f}%")
    c3.metric("Model Used",  model_type.upper())

    # ── Location map ─────────────────────────────────────────────────────────
    st.subheader("🌍 Predicted Location")

    pred_df = pd.DataFrame({"lat": [lat_abs], "lon": [lon], "risk": [result["prediction"]]})

    fig_pred = px.scatter_mapbox(
        pred_df, lat="lat", lon="lon", color="risk", zoom=4, height=450,
        color_discrete_map={
            "Very Low": "green", "Low": "lightgreen",
            "Medium": "orange",  "High": "red", "Very High": "darkred"
        }
    )
    fig_pred.update_layout(mapbox_style="open-street-map")
    fig_pred.update_traces(marker=dict(size=18))
    st.plotly_chart(fig_pred, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("GeoHazard ML Project | CAT II Assessment")
