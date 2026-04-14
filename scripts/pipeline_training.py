# ==========================================
# GEO HAZARD — SCIKIT-LEARN PIPELINE
# Rubric requirement: production-ready
# pipeline with scaling + modeling.
# ==========================================

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score,
                              roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------------------------------
# 1. LOAD & PREPARE DATA
# ------------------------------------------
df = pd.read_csv("data/processed/featured_dataset.csv")

drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid', 'inform_risk_score']
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df['lat_seismic_zone']  = np.where(df['lat_abs'] <= 60, 1, 0)
df['seismic_intensity'] = df['seismic_energy_proxy'] * df['eq_frequency_score']
df['weighted_mag']      = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)
df = pd.get_dummies(df, columns=['continent'], drop_first=True)

X = df.drop('risk_label', axis=1)
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE before pipeline (pipeline handles scaling internally here)
smote = SMOTE(
    sampling_strategy={
        'Very Low':  1000,
        'Very High': int(y_train.value_counts()['Very High']),
        'High':      int(y_train.value_counts()['High']),
        'Medium':    int(y_train.value_counts()['Medium']),
        'Low':       int(y_train.value_counts()['Low']),
    },
    random_state=42, k_neighbors=3
)
X_train_r, y_train_r = smote.fit_resample(X_train, y_train)

print(f"Training shape after SMOTE: {X_train_r.shape}")


# ------------------------------------------
# 2. PIPELINE 1 — Logistic Regression (Baseline)
# ------------------------------------------
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

pipe_lr.fit(X_train_r, y_train_r)
y_pred_lr = pipe_lr.predict(X_test)

print("\n=== Pipeline: Logistic Regression (Baseline) ===")
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_lr, average='macro'):.4f}")


# ------------------------------------------
# 3. PIPELINE 2 — Random Forest
# ------------------------------------------
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        random_state=42, n_jobs=-1, class_weight='balanced'
    ))
])

pipe_rf.fit(X_train_r, y_train_r)
y_pred_rf = pipe_rf.predict(X_test)

print("\n=== Pipeline: Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_rf, average='macro'):.4f}")


# ------------------------------------------
# 4. PIPELINE 3 — XGBoost (Best Model)
# ------------------------------------------
le = LabelEncoder()
le.fit(y)
y_train_enc = le.transform(y_train_r)
y_test_enc  = le.transform(y_test)

pipe_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    ))
])

pipe_xgb.fit(X_train_r, y_train_enc)
y_pred_xgb_enc = pipe_xgb.predict(X_test)
y_pred_xgb     = le.inverse_transform(y_pred_xgb_enc)

print("\n=== Pipeline: XGBoost (Best Model) ===")
print(classification_report(y_test, y_pred_xgb))
print(f"Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")


# ------------------------------------------
# 5. COMPARISON SUMMARY
# ------------------------------------------
print("\n" + "="*60)
print("COMPARATIVE ANALYSIS SUMMARY")
print("="*60)
print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
print("-"*60)
for name, y_true, y_pred in [
    ("Logistic Regression (Baseline)", y_test,     y_pred_lr),
    ("Random Forest",                  y_test,     y_pred_rf),
    ("XGBoost (Best)",                 y_test,     y_pred_xgb),
]:
    print(f"{name:<30} {accuracy_score(y_true,y_pred):>10.4f} {f1_score(y_true,y_pred,average='macro'):>10.4f}")
print("="*60)


# ------------------------------------------
# 6. SAVE PIPELINES
# ------------------------------------------
joblib.dump(pipe_lr,  "models/pipeline_lr.pkl")
joblib.dump(pipe_rf,  "models/pipeline_rf.pkl")
joblib.dump(pipe_xgb, "models/pipeline_xgb.pkl")
joblib.dump(le,       "models/label_encoder.pkl")

print("\n✅ All pipelines saved.")
