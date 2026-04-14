# ==========================================
# GEO HAZARD — XGBOOST MODEL TRAINING
# NOTE: Run AFTER rf_model_training.py
#       (reuses the same scaler + SMOTE data)
# ==========================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ------------------------------------------
# 1. LOAD & PREPARE DATA  (same as RF script)
# ------------------------------------------
df = pd.read_csv("data/processed/featured_dataset.csv")

drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid', 'inform_risk_score']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)

df['lat_seismic_zone'] = np.where(df['lat_abs'] <= 60, 1, 0)
df['seismic_intensity'] = df['seismic_energy_proxy'] * df['eq_frequency_score']
df['weighted_mag']      = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)

df = pd.get_dummies(df, columns=['continent'], drop_first=True)

X = df.drop('risk_label', axis=1)
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------
# 2. LOAD SHARED SCALER (saved by RF script)
# ------------------------------------------
scaler = joblib.load("models/scaler.pkl")
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)
print("Loaded shared scaler.")


# ------------------------------------------
# 3. SMOTE  (same strategy as RF)
# ------------------------------------------
smote = SMOTE(
    sampling_strategy={
        'Very Low':  1000,
        'Very High': int(y_train.value_counts()['Very High']),
        'High':      int(y_train.value_counts()['High']),
        'Medium':    int(y_train.value_counts()['Medium']),
        'Low':       int(y_train.value_counts()['Low']),
    },
    random_state=42,
    k_neighbors=3
)
X_train_s, y_train_r = smote.fit_resample(X_train_s, y_train)


# ------------------------------------------
# 4. ENCODE TARGET  (required for XGBoost)
# ------------------------------------------
le = LabelEncoder()
le.fit(y)   # fit on full y so all classes are known
y_train_enc = le.transform(y_train_r)
y_test_enc  = le.transform(y_test)
print("Label classes:", le.classes_)


# ------------------------------------------
# 5. TRAIN XGBOOST
# ------------------------------------------
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb.fit(X_train_s, y_train_enc)
print("\n✅ XGBoost training complete!")


# ------------------------------------------
# 6. EVALUATION
# ------------------------------------------
y_pred = xgb.predict(X_test_s)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_enc, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
y_test_bin = label_binarize(y_test_enc, classes=[0,1,2,3,4])
roc_auc = roc_auc_score(y_test_bin, xgb.predict_proba(X_test_s), multi_class='ovr', average='macro')
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Macro ROC-AUC: {roc_auc:.4f}")


# ------------------------------------------
# 7. SAVE
# ------------------------------------------
joblib.dump(xgb, "models/xgb_model.pkl")
joblib.dump(le,  "models/label_encoder.pkl")

print("\n✅ XGB model and label encoder saved.")
