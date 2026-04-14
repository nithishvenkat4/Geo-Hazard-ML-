# ==========================================
# GEO HAZARD ML — GRADIO UI (Hugging Face)
# ==========================================

import os, sys
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR  = os.path.join(ROOT, "data", "processed")

# ── Load models ───────────────────────────────────────────────────────────────
rf_model        = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model       = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
label_encoder   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ── Constants ─────────────────────────────────────────────────────────────────
CONTINENTS  = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
RISK_EMOJI  = {"Very Low": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Very High": "🚨"}
RISK_COLOR  = {"Very Low": "#2ecc71", "Low": "#f1c40f", "Medium": "#e67e22",
               "High": "#e74c3c", "Very High": "#8e1010"}
RISK_ORDER  = ["Very Low", "Low", "Medium", "High", "Very High"]


# ── Feature engineering (mirrors training) ────────────────────────────────────
def engineer_and_predict(eq_count, eq_mag_mean, eq_depth_mean, has_seismic,
                          eq_freq, seismic_energy, mag_depth_ratio,
                          depth_cat, lat_abs, lon, neighbor_eq, continent, model_choice):

    raw = {
        "eq_count":              float(eq_count),
        "eq_mag_mean":           float(eq_mag_mean),
        "eq_depth_mean":         float(eq_depth_mean),
        "has_seismic_activity":  float(has_seismic),
        "eq_frequency_score":    float(eq_freq),
        "seismic_energy_proxy":  float(seismic_energy),
        "mag_depth_ratio":       float(mag_depth_ratio),
        "depth_category":        float(str(depth_cat)[0]),  # extract leading digit
        "lat_abs":               float(abs(lat_abs)),
        "neighbor_eq_count":     float(neighbor_eq),
        "continent":             continent,
    }

    df = pd.DataFrame([raw])
    df["lat_seismic_zone"]  = np.where(df["lat_abs"] <= 60, 1, 0)
    df["seismic_intensity"] = df["seismic_energy_proxy"] * df["eq_frequency_score"]
    df["weighted_mag"]      = df["eq_mag_mean"] / (df["eq_depth_mean"] + 1)
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(df)
    model_key = "rf" if model_choice == "Random Forest" else "xgb"

    if model_key == "rf":
        pred    = rf_model.predict(X_scaled)[0]
        proba   = rf_model.predict_proba(X_scaled)[0]
        classes = rf_model.classes_
    else:
        enc     = xgb_model.predict(X_scaled)[0]
        pred    = label_encoder.inverse_transform([enc])[0]
        proba   = xgb_model.predict_proba(X_scaled)[0]
        classes = label_encoder.classes_

    confidence = float(max(proba))
    icon  = RISK_EMOJI.get(pred, "⚪")
    color = RISK_COLOR.get(pred, "#aaa")

    # ── Result card ───────────────────────────────────────────────────────────
    result_html = f"""
    <div style="background:#1e2130;border-radius:14px;padding:28px 32px;text-align:center;
                border:2px solid {color};max-width:480px;margin:auto;">
      <div style="font-size:52px;margin-bottom:6px">{icon}</div>
      <div style="color:#aaa;font-size:13px;letter-spacing:2px;text-transform:uppercase">Risk Level</div>
      <div style="color:{color};font-size:38px;font-weight:800;margin:6px 0">{pred}</div>
      <div style="color:#ccc;font-size:14px;margin-top:4px">
        Confidence: <strong style="color:white">{confidence*100:.1f}%</strong>
        &nbsp;|&nbsp; Model: <strong style="color:white">{model_choice}</strong>
      </div>
      <div style="margin-top:18px;background:#0f1117;border-radius:8px;overflow:hidden;height:12px">
        <div style="width:{confidence*100:.1f}%;height:100%;background:{color}"></div>
      </div>
    </div>"""

    # ── Probability bar chart ─────────────────────────────────────────────────
    prob_dict     = dict(zip(classes, proba))
    ordered_probs = [prob_dict.get(c, 0) for c in RISK_ORDER]
    bar_colors    = [RISK_COLOR[c] for c in RISK_ORDER]

    fig_prob = go.Figure(go.Bar(
        x=RISK_ORDER, y=ordered_probs,
        marker_color=bar_colors,
        text=[f"{p*100:.1f}%" for p in ordered_probs],
        textposition="outside", textfont=dict(color="white", size=11)
    ))
    fig_prob.update_layout(
        title=dict(text="Class Probabilities", font=dict(color="white", size=14)),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", range=[0, 1.1]),
        margin=dict(t=40, b=10, l=10, r=10), height=280
    )

    # ── Location map ─────────────────────────────────────────────────────────
    fig_map = go.Figure(go.Scattermapbox(
        lat=[lat_abs], lon=[lon],
        mode="markers",
        marker=dict(size=18, color=color, opacity=0.9),
        text=[f"<b>{pred}</b><br>Confidence: {confidence*100:.1f}%"],
        hoverinfo="text"
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=lat_abs, lon=lon), zoom=4),
        paper_bgcolor="#0f1117",
        margin=dict(t=0, b=0, l=0, r=0), height=340
    )

    return result_html, fig_prob, fig_map


# ── Global heatmap (loaded once) ──────────────────────────────────────────────
def build_heatmap():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"))
        fig = px.density_map(
            df, lat="lat_grid", lon="lon_grid", z="inform_risk_score",
            radius=14, center=dict(lat=20, lon=0), zoom=1,
            color_continuous_scale="YlOrRd",
            title="Global INFORM Risk Score Heatmap",
            labels={"inform_risk_score": "Risk Score"}
        )
        fig.update_layout(
            map_style="carto-darkmatter",
            paper_bgcolor="#0f1117", font=dict(color="white"),
            margin=dict(t=40, b=0, l=0, r=0), height=440,
            title_font=dict(size=15)
        )
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"Map unavailable: {e}")


HEATMAP = build_heatmap()


# ── Auto-compute derived fields ───────────────────────────────────────────────
def auto_compute(eq_count, eq_mag_mean, eq_depth_mean):
    freq   = round(float(np.log1p(eq_count)), 4)
    energy = round(float(10 ** (1.5 * eq_mag_mean)), 2)
    ratio  = round(float(eq_mag_mean / (eq_depth_mean + 1)), 4)
    return freq, energy, ratio


# ── Gradio UI ─────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="orange", secondary_hue="slate",
    neutral_hue="slate", font=gr.themes.GoogleFont("Inter")
)
CSS = """
body, .gradio-container { background:#0f1117 !important; }
.gr-panel, .gr-box       { background:#1e2130 !important; border:1px solid #2a2a3a !important; }
label, .gr-block-label   { color:#cccccc !important; }
h1,h2,h3                 { color:white !important; }
.gr-button               { border-radius:10px !important; }
"""

with gr.Blocks(title="GeoHazard ML") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:2.2rem;font-weight:800;color:white;margin:0">
        🌍 GeoHazard ML
      </h1>
      <p style="color:#8892a4;font-size:1rem;margin:6px 0 0">
        Geological Hazard Risk Classification &nbsp;·&nbsp;
        USGS + INFORM &nbsp;·&nbsp; CAT II Assessment
      </p>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: Global heatmap ─────────────────────────────────────────────
        with gr.Tab("🗺️ Global Risk Map"):
            gr.Plot(value=HEATMAP, label="")
            gr.HTML("<p style='color:#555;font-size:12px;text-align:center'>"
                    "INFORM Risk Score across 14,613 global 1°×1° grid cells "
                    "· USGS Earthquake Catalog + INFORM Risk Index</p>")

        # ── Tab 2: Prediction ─────────────────────────────────────────────────
        with gr.Tab("🔍 Predict Risk"):
            gr.HTML("<p style='color:#8892a4;margin:0 0 16px'>"
                    "Enter seismic parameters for a location to predict "
                    "its INFORM risk category.</p>")

            with gr.Row():

                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:#e67e22;margin:0 0 10px'>📡 Seismic Parameters</h3>")
                    eq_count      = gr.Number(label="Earthquake Count",        value=3,    precision=0, minimum=0)
                    eq_mag_mean   = gr.Number(label="Avg Magnitude (Mw)",       value=4.8,  step=0.1)
                    eq_depth_mean = gr.Number(label="Avg Depth (km)",            value=15.0, step=1.0, minimum=0)
                    has_seismic   = gr.Radio(label="Seismic Activity Present",
                                             choices=[(1, "Yes"), (0, "No")], value=1)
                    depth_cat     = gr.Dropdown(
                        label="Depth Category",
                        choices=["0 — No data",
                                 "1 — Shallow (< 70 km)",
                                 "2 — Intermediate (70–300 km)",
                                 "3 — Deep (> 300 km)"],
                        value="1 — Shallow (< 70 km)"
                    )
                    neighbor_eq   = gr.Number(label="Neighbour EQ Count (smoothed)",
                                              value=2.0, step=0.1, minimum=0)

                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:#3498db;margin:0 0 10px'>⚡ Derived Features <span style='font-size:12px;color:#555'>(auto-filled)</span></h3>")
                    eq_freq        = gr.Number(label="EQ Frequency Score",   value=1.386, step=0.01)
                    seismic_energy = gr.Number(label="Seismic Energy Proxy", value=1778.3, step=1.0)
                    mag_depth_r    = gr.Number(label="Mag / Depth Ratio",    value=0.3,   step=0.01)
                    gr.HTML("<div style='height:10px'></div>")
                    gr.HTML("<h3 style='color:#2ecc71;margin:0 0 10px'>📍 Location</h3>")
                    lat_abs   = gr.Slider(label="Latitude  (abs, 0–90)",
                                          minimum=0, maximum=90, step=0.5, value=12.0)
                    lon       = gr.Slider(label="Longitude (-180 to 180)",
                                          minimum=-180, maximum=180, step=0.5, value=78.0)
                    continent = gr.Dropdown(label="Continent", choices=CONTINENTS, value="Asia")

                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:#9b59b6;margin:0 0 10px'>🤖 Model</h3>")
                    model_choice = gr.Radio(
                        label="Choose Model",
                        choices=["XGBoost (Best)", "Random Forest"],
                        value="XGBoost (Best)"
                    )
                    predict_btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")
                    gr.HTML("<div style='height:12px'></div>")
                    result_html = gr.HTML(label="")

            with gr.Row():
                fig_prob = gr.Plot(label="Class Probabilities")
                fig_map  = gr.Plot(label="Location Map")

            # Auto-fill derived fields
            for inp in [eq_count, eq_mag_mean, eq_depth_mean]:
                inp.change(fn=auto_compute,
                           inputs=[eq_count, eq_mag_mean, eq_depth_mean],
                           outputs=[eq_freq, seismic_energy, mag_depth_r])

            predict_btn.click(
                fn=engineer_and_predict,
                inputs=[eq_count, eq_mag_mean, eq_depth_mean, has_seismic,
                        eq_freq, seismic_energy, mag_depth_r, depth_cat,
                        lat_abs, lon, neighbor_eq, continent, model_choice],
                outputs=[result_html, fig_prob, fig_map]
            )

        # ── Tab 3: Model performance ──────────────────────────────────────────
        with gr.Tab("📊 Model Performance"):
            gr.HTML("""
            <div style="padding:16px">
            <h2 style="color:white">Comparative Analysis</h2>
            <table style="width:100%;border-collapse:collapse;color:#ccc;font-size:14px">
              <thead>
                <tr style="background:#2a2a3a;color:white">
                  <th style="padding:10px 16px;text-align:left">Model</th>
                  <th style="padding:10px 16px;text-align:center">Accuracy</th>
                  <th style="padding:10px 16px;text-align:center">Macro F1</th>
                  <th style="padding:10px 16px;text-align:center">ROC-AUC</th>
                  <th style="padding:10px 16px;text-align:center">Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom:1px solid #2a2a3a">
                  <td style="padding:10px 16px">Logistic Regression</td>
                  <td style="text-align:center">58.3%</td>
                  <td style="text-align:center">0.477</td>
                  <td style="text-align:center">0.906</td>
                  <td style="text-align:center;color:#888">Baseline</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a3a">
                  <td style="padding:10px 16px">Random Forest</td>
                  <td style="text-align:center">76.3%</td>
                  <td style="text-align:center">0.644</td>
                  <td style="text-align:center">0.959</td>
                  <td style="text-align:center;color:#3498db">+18.0% over baseline</td>
                </tr>
                <tr style="background:#1a2a1a">
                  <td style="padding:10px 16px;color:#2ecc71;font-weight:700">✅ XGBoost (Best)</td>
                  <td style="text-align:center;color:#2ecc71;font-weight:700">80.1%</td>
                  <td style="text-align:center;color:#2ecc71;font-weight:700">0.674</td>
                  <td style="text-align:center;color:#2ecc71;font-weight:700">0.961</td>
                  <td style="text-align:center;color:#2ecc71">+21.8% over baseline</td>
                </tr>
              </tbody>
            </table>

            <h3 style="color:white;margin-top:28px">Dataset</h3>
            <ul style="color:#aaa;line-height:1.9">
              <li>📡 <strong style="color:white">USGS Earthquake Catalog</strong> — 166,293 events, M4.0+, 2000–2023</li>
              <li>🌐 <strong style="color:white">INFORM Risk Index</strong> — 191 countries, EU Joint Research Centre</li>
              <li>🗺️ <strong style="color:white">14,613 grid cells</strong> — 1°×1° global resolution with spatial join</li>
              <li>⚖️ <strong style="color:white">Class imbalance handled</strong> — SMOTE (Very Low boosted 5×)</li>
            </ul>

            <h3 style="color:white;margin-top:20px">Features Used (18 total)</h3>
            <ul style="color:#aaa;line-height:1.9">
              <li>EQ Count · Avg Magnitude · Avg Depth · Seismic Activity flag</li>
              <li>Frequency Score · Energy Proxy · Mag/Depth Ratio · Depth Category</li>
              <li>Neighbour EQ Count (Gaussian spatial smoothing)</li>
              <li>Absolute Latitude · Seismic Zone Flag · Continent (one-hot encoded)</li>
              <li>Seismic Intensity · Weighted Magnitude <em>(engineered in Phase A)</em></li>
            </ul>
            </div>""")

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px;color:#444;font-size:12px">
      GeoHazard ML · CAT II Academic Assessment · Data: USGS + INFORM Risk Index
    </div>""")


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
