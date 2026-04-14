# ==========================================
# GEO HAZARD — RF MODEL TRAINING PIPELINE
# ==========================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE


# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("data/processed/featured_dataset.csv")
print("Loaded Shape:", df.shape)


# ------------------------------------------
# 2. DROP NON-ML COLUMNS
# ------------------------------------------
drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid', 'inform_risk_score']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)
print("After Dropping Columns:", df.shape)


# ------------------------------------------
# 3. ADD ENGINEERED FEATURES
# ------------------------------------------
df['lat_seismic_zone'] = np.where(df['lat_abs'] <= 60, 1, 0)
df['seismic_intensity'] = df['seismic_energy_proxy'] * df['eq_frequency_score']
df['weighted_mag']      = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)


# ------------------------------------------
# 4. ENCODE CATEGORICAL FEATURES
# ------------------------------------------
df = pd.get_dummies(df, columns=['continent'], drop_first=True)
print("After Encoding:", df.shape)


# ------------------------------------------
# 5. SPLIT FEATURES & TARGET
# ------------------------------------------
X = df.drop('risk_label', axis=1)
y = df['risk_label']
print("Feature Shape:", X.shape)


# ------------------------------------------
# 6. TRAIN-TEST SPLIT (STRATIFIED)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ------------------------------------------
# 7. FEATURE SCALING  (shared scaler — used by both RF and XGB)
# ------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("\nScaling Complete")


# ------------------------------------------
# 8. HANDLE CLASS IMBALANCE (SMOTE)
# ------------------------------------------
print("\nBefore SMOTE:\n", y_train.value_counts().to_dict())

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
print("After SMOTE:", dict(pd.Series(y_train_r).value_counts()))
print("New Training Shape:", X_train_s.shape)


# ------------------------------------------
# 9. TRAIN RANDOM FOREST
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train_s, y_train_r)
print("\n✅ Random Forest training complete!")


# ------------------------------------------
# 10. EVALUATION
# ------------------------------------------
y_pred = rf.predict(X_test_s)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred, labels=['Very Low','Low','Medium','High','Very High']))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
y_test_bin = label_binarize(y_test, classes=rf.classes_)
roc_auc = roc_auc_score(y_test_bin, rf.predict_proba(X_test_s), multi_class='ovr', average='macro')
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Macro ROC-AUC: {roc_auc:.4f}")


# ------------------------------------------
# 11. SAVE
# ------------------------------------------
joblib.dump(rf,                    "models/rf_model.pkl")
joblib.dump(scaler,                "models/scaler.pkl")        # shared scaler
joblib.dump(X.columns.tolist(),    "models/feature_columns.pkl")

print("\n✅ RF model, scaler, and feature columns saved.")
