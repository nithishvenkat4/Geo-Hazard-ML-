# ==========================================
# GEO HAZARD — PREDICTION ENGINE (IMPROVED)
# ==========================================

import pandas as pd
import joblib
import os

# ------------------------------------------
# PATH HANDLING (IMPORTANT)
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Label mapping (if needed)
label_map = {
    0: "High",
    1: "Low",
    2: "Medium",
    3: "Very High",
    4: "Very Low"
}


# ------------------------------------------
# CORE FUNCTION
# ------------------------------------------
def predict(input_data: dict, model="rf"):
    df = pd.DataFrame([input_data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Model selection
    if model == "rf":
        pred = rf_model.predict(df_scaled)[0]
        prob = max(rf_model.predict_proba(df_scaled)[0])

    elif model == "xgb":
        pred_encoded = xgb_model.predict(df_scaled)[0]
        pred = label_map[pred_encoded]

        prob = max(xgb_model.predict_proba(df_scaled)[0])

    else:
        raise ValueError("Model must be 'rf' or 'xgb'")

    return {
        "prediction": pred,
        "confidence": round(float(prob), 3)
    }
if __name__ == "__main__":

    sample_input = {
        "eq_count": 3,
        "eq_mag_mean": 4.8,
        "eq_depth_mean": 15,
        "has_seismic_activity": 1,
        "eq_frequency_score": 1.3,
        "seismic_energy_proxy": 1500,
        "mag_depth_ratio": 0.3,
        "depth_category": 1,
        "lat_abs": 12,
        "neighbor_eq_count": 2,
        "continent_Asia": 1
    }

    print("RF:", predict(sample_input, model="rf"))
    print("XGB:", predict(sample_input, model="xgb"))