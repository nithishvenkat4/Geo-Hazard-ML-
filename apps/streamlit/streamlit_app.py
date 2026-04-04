import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
from core.predict import predict
# --------------------------------------
# IMPORT BACKEND
# --------------------------------------

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="Geo Hazard Prediction", layout="wide")

# --------------------------------------
# TITLE
# --------------------------------------
st.title("🌍 Geo Hazard Prediction System")

st.markdown("""
Analyze geological hazard risk using ML models (Random Forest & XGBoost).  
Visualize both global context and location-specific predictions.
""")

# --------------------------------------
# MODEL SELECTION
# --------------------------------------
st.subheader("⚙️ Model Selection")

model_choice = st.radio(
    "Choose Model:",
    ["Random Forest", "XGBoost"]
)

model_type = "rf" if model_choice == "Random Forest" else "xgb"

# --------------------------------------
# SECTION 1: HISTORICAL MAP (STATIC)
# --------------------------------------
# --------------------------------------
# HISTORICAL HEATMAP (REAL DATA)
# --------------------------------------
st.subheader("🌍 Historical Hazard Heatmap")

df = pd.read_csv("data/processed/grid_dataset.csv")

fig_hist = px.density_mapbox(
    df,
    lat="lat_grid",
    lon="lon_grid",
    z="inform_risk_score",   # 🔥 core signal
    radius=6,
    center=dict(lat=20, lon=0),
    zoom=1,
    height=500,
    color_continuous_scale="YlOrRd"
)
fig_hist.update_traces(
    hovertemplate=
    "<b>Country:</b> %{customdata[0]}<br>" +
    "<b>Risk Score:</b> %{z}<extra></extra>",
    customdata=df[["country"]]
)
fig_hist.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_hist, use_container_width=True)
# --------------------------------------
# SECTION 2: INPUT FORM
# --------------------------------------
st.subheader("🧾 Enter Parameters for Prediction")

col1, col2 = st.columns(2)

with col1:
    eq_count = st.number_input("Earthquake Count", min_value=0, value=1)
    eq_mag_mean = st.number_input("Avg Magnitude", value=4.5)
    eq_depth_mean = st.number_input("Avg Depth", value=10.0)

    has_seismic_activity = st.selectbox("Seismic Activity", [0, 1])
    eq_frequency_score = st.number_input("Frequency Score", value=1.0)

with col2:
    seismic_energy_proxy = st.number_input("Seismic Energy", value=1000.0)
    mag_depth_ratio = st.number_input("Magnitude-Depth Ratio", value=0.2)
    depth_category = st.selectbox("Depth Category", [0, 1, 2])

    lat_abs = st.number_input("Latitude", value=10.0)
    lon_abs = st.number_input("Longitude", value=78.0)

    neighbor_eq_count = st.number_input("Neighbor Earthquakes", value=1)

continent = st.selectbox(
    "Continent",
    ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"]
)

predict_btn = st.button("🔍 Predict Risk")

# --------------------------------------
# SECTION 3: RESULT + DYNAMIC MAP
# --------------------------------------
if predict_btn:

    # Prepare input
    input_data = {
        "eq_count": eq_count,
        "eq_mag_mean": eq_mag_mean,
        "eq_depth_mean": eq_depth_mean,
        "has_seismic_activity": has_seismic_activity,
        "eq_frequency_score": eq_frequency_score,
        "seismic_energy_proxy": seismic_energy_proxy,
        "mag_depth_ratio": mag_depth_ratio,
        "depth_category": depth_category,
        "lat_abs": lat_abs,
        "neighbor_eq_count": neighbor_eq_count,
    }

    input_data[f"continent_{continent}"] = 1

    # Predict
    result = predict(input_data, model=model_type)

    # --------------------------------------
    # RESULT DISPLAY
    # --------------------------------------
    st.subheader("📊 Prediction Result")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Risk Level", result["prediction"])

    with col4:
        st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    # --------------------------------------
    # DYNAMIC MAP
    # --------------------------------------
    st.subheader("🌍 Predicted Location")

    pred_df = pd.DataFrame({
        "lat": [lat_abs],
        "lon": [lon_abs],
        "risk": [result["prediction"]]
    })

    fig_pred = px.scatter_mapbox(
    pred_df,
    lat="lat",
    lon="lon",
    color="risk",
    zoom=3,
    height=500,
    color_discrete_map={
        "Very Low": "green",
        "Low": "lightgreen",
        "Medium": "orange",
        "High": "red",
        "Very High": "darkred"
    }
)

    fig_pred.update_layout(mapbox_style="open-street-map")
    fig_pred.update_traces(marker=dict(size=18))

    st.plotly_chart(fig_pred, use_container_width=True)

# --------------------------------------
# FOOTER
# --------------------------------------
st.markdown("---")
st.caption("Geo Hazard ML Project | Streamlit Dashboard")