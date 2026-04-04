from flask import Blueprint, render_template, request
from core.predict import predict

import pandas as pd
import plotly.express as px
data = pd.read_csv("data/processed/master_dataset.csv")
import plotly.express as px

def create_heatmap():
    fig = px.density_mapbox(
        data,
        lat="lat_grid",
        lon="lon_grid",
        z="inform_risk_score",
        radius=12,  # 🔥 smoother heat
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
    return render_template("index.html", map_html=map_html)

@main.route("/predict", methods=["POST"])
def make_prediction():
    try:
        form_data = request.form.to_dict()

        # Extract model choice
        model_choice = form_data.pop("model")

        # Convert all values to float
        input_data = {k: float(v) for k, v in form_data.items()}

        result = predict(input_data, model=model_choice)

        return render_template(
    "result.html",
    prediction=result["prediction"],
    confidence=result["confidence"],
    model=model_choice.upper(),
    lat=input_data.get("lat_abs", 0),
    lon=input_data.get("lon", 0)  # we’ll add this field
)

    except Exception as e:
        return f"Error: {str(e)}"