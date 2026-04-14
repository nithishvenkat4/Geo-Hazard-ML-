# ==========================================
# GEO HAZARD — PREDICTION ENGINE
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------
# PATH HANDLING
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

rf_model        = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model       = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))   # shared scaler
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
label_encoder   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


# ------------------------------------------
# FEATURE ENGINEERING (must match training)
# ------------------------------------------
def engineer_features(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    # Derived features added during Phase A retraining
    df['lat_seismic_zone'] = np.where(df['lat_abs'] <= 60, 1, 0)
    df['seismic_intensity'] = df['seismic_energy_proxy'] * df['eq_frequency_score']
    df['weighted_mag']      = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)

    # One-hot encode continent
    df = pd.get_dummies(df)

    # Align to training columns, fill missing with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ------------------------------------------
# CORE PREDICTION FUNCTION
# ------------------------------------------
def predict(input_data: dict, model: str = "rf") -> dict:
    """
    Parameters
    ----------
    input_data : dict
        Must include: eq_count, eq_mag_mean, eq_depth_mean,
        has_seismic_activity, eq_frequency_score, seismic_energy_proxy,
        mag_depth_ratio, depth_category, lat_abs, neighbor_eq_count,
        continent (string, e.g. 'Asia')
    model : str
        'rf' for Random Forest, 'xgb' for XGBoost

    Returns
    -------
    dict with keys: prediction (str), confidence (float)
    """
    df = engineer_features(input_data)
    df_scaled = scaler.transform(df)

    if model == "rf":
        pred  = rf_model.predict(df_scaled)[0]
        prob  = max(rf_model.predict_proba(df_scaled)[0])

    elif model == "xgb":
        pred_enc = xgb_model.predict(df_scaled)[0]
        pred     = label_encoder.inverse_transform([pred_enc])[0]
        prob     = max(xgb_model.predict_proba(df_scaled)[0])

    else:
        raise ValueError("model must be 'rf' or 'xgb'")

    return {
        "prediction": str(pred),
        "confidence": round(float(prob), 3)
    }


# ------------------------------------------
# QUICK TEST
# ------------------------------------------
if __name__ == "__main__":
    sample = {
        "eq_count":             3,
        "eq_mag_mean":          4.8,
        "eq_depth_mean":        15.0,
        "has_seismic_activity": 1,
        "eq_frequency_score":   1.3,
        "seismic_energy_proxy": 1500.0,
        "mag_depth_ratio":      0.3,
        "depth_category":       1,
        "lat_abs":              12.0,
        "neighbor_eq_count":    2.0,
        "continent":            "Asia"
    }

    print("RF  :", predict(sample, model="rf"))
    print("XGB :", predict(sample, model="xgb"))