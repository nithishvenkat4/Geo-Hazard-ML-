from flask import Blueprint, render_template, request
from core.predict import predict
import pandas as pd
import plotly.express as px

data = pd.read_csv("data/processed/master_dataset.csv")

CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

def create_heatmap():
    fig = px.density_mapbox(
        data,
        lat="lat_grid",
        lon="lon_grid",
        z="inform_risk_score",
        radius=12,
        center=dict(lat=20, lon=0),
        zoom=1,
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Risk Score")
    )
    return fig.to_html(full_html=False)


main = Blueprint("main", __name__)

@main.route("/")
def home():
    map_html = create_heatmap()
    return render_template("index.html", map_html=map_html, continents=CONTINENTS)

@main.route("/predict", methods=["POST"])
def make_prediction():
    try:
        form = request.form

        # Build input dict — continent passed as string, predict.py handles encoding
        input_data = {
            "eq_count":             float(form["eq_count"]),
            "eq_mag_mean":          float(form["eq_mag_mean"]),
            "eq_depth_mean":        float(form["eq_depth_mean"]),
            "has_seismic_activity": float(form["has_seismic_activity"]),
            "eq_frequency_score":   float(form["eq_frequency_score"]),
            "seismic_energy_proxy": float(form["seismic_energy_proxy"]),
            "mag_depth_ratio":      float(form["mag_depth_ratio"]),
            "depth_category":       float(form["depth_category"]),
            "lat_abs":              float(form["lat_abs"]),
            "neighbor_eq_count":    float(form["neighbor_eq_count"]),
            "continent":            form["continent"],   # string — e.g. "Asia"
        }

        model_choice = form["model"]
        result = predict(input_data, model=model_choice)

        # Pass actual lat for map; lon comes from form too
        lat = input_data["lat_abs"]
        lon = float(form.get("lon", 0))

        return render_template(
            "result.html",
            prediction=result["prediction"],
            confidence=result["confidence"],
            model=model_choice.upper(),
            lat=lat,
            lon=lon
        )

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>← Back</a>"
