# ==========================================
# GEO HAZARD — MODEL TRAINING PIPELINE
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("featured_dataset.csv")

print("Loaded Shape:", df.shape)


# ------------------------------------------
# 2. DROP NON-ML COLUMNS
# ------------------------------------------
drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid']
df = df.drop(columns=['inform_risk_score']) 
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)

print("After Dropping Columns:", df.shape)


# ------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# ------------------------------------------
df = pd.get_dummies(df, columns=['continent'], drop_first=True)

print("After Encoding:", df.shape)


# ------------------------------------------
# 4. SPLIT FEATURES & TARGET
# ------------------------------------------
X = df.drop('risk_label', axis=1)
y = df['risk_label']

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ------------------------------------------
# 5. TRAIN-TEST SPLIT (STRATIFIED)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ------------------------------------------
# 6. FEATURE SCALING
# ------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Complete")


# ------------------------------------------
# 7. HANDLE CLASS IMBALANCE (SMOTE)
# ------------------------------------------
print("\nBefore SMOTE:\n")
print(pd.Series(y_train).value_counts())

smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n")
print(pd.Series(y_train).value_counts())
print("\nNew Training Shape:", X_train.shape)


# ------------------------------------------
# 8. TRAIN RANDOM FOREST MODEL
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,              # ← IMPORTANT CHANGE
    min_samples_split=5,       # ← stabilizes splits
    min_samples_leaf=2,        # ← reduces noise learning
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

print("\n✅ Random Forest training complete!")

# ==========================================
# STEP 14 — MODEL EVALUATION
# ==========================================

from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------
# 1. PREDICT ON TEST DATA (IMPORTANT)
# ------------------------------------------
y_pred = rf.predict(X_test)

# ------------------------------------------
# 2. CONFUSION MATRIX
# ------------------------------------------
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------
# 3. CLASSIFICATION REPORT
# ------------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Random Forest model saved!")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")