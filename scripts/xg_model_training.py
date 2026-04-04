# ==========================================
# GEO HAZARD — XGBOOST MODEL TRAINING
# ==========================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("featured_dataset.csv")

print("Loaded Shape:", df.shape)


# ------------------------------------------
# 2. DROP NON-ML COLUMNS + LEAKAGE COLUMN
# ------------------------------------------
drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid', 'inform_risk_score']

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


# ------------------------------------------
# 5. ENCODE TARGET (REQUIRED FOR XGBOOST)
# ------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y)


# ------------------------------------------
# 6. TRAIN-TEST SPLIT
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
# 7. FEATURE SCALING
# ------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Complete")


# ------------------------------------------
# 8. HANDLE CLASS IMBALANCE (SMOTE)
# ------------------------------------------
print("\nBefore SMOTE:\n")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n")
print(pd.Series(y_train).value_counts())
print("\nNew Training Shape:", X_train.shape)


# ------------------------------------------
# 9. TRAIN XGBOOST MODEL
# ------------------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb.fit(X_train, y_train)

print("\n✅ XGBoost training complete!")


# ------------------------------------------
# 10. EVALUATION
# ------------------------------------------
y_pred = xgb.predict(X_test)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler_xgb.pkl")

print("✅ XGBoost model saved!")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")