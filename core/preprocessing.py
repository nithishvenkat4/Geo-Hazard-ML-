# ==========================================
# STEP 11 — PREPROCESSING PIPELINE
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# 1. LOAD FEATURED DATASET
# ------------------------------------------
df = pd.read_csv("featured_dataset.csv")

print("Loaded Shape:", df.shape)


# ------------------------------------------
# 2. DROP NON-ML COLUMNS
# ------------------------------------------
drop_cols = ['country', 'iso3', 'lat_grid', 'lon_grid']

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)

print("After Dropping Columns:", df.shape)


# ------------------------------------------
# 3. ONE-HOT ENCODE CATEGORICAL FEATURES
# ------------------------------------------
df = pd.get_dummies(df, columns=['continent'], drop_first=True)

print("After Encoding:", df.shape)


# ------------------------------------------
# 4. SEPARATE FEATURES & TARGET
# ------------------------------------------
X = df.drop('risk_label', axis=1)
y = df['risk_label']

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ------------------------------------------
# 5. STRATIFIED TRAIN-TEST SPLIT
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
# 6. VERIFY CLASS DISTRIBUTION
# ------------------------------------------
print("\nOriginal Distribution:\n", y.value_counts(normalize=True))
print("\nTrain Distribution:\n", y_train.value_counts(normalize=True))
print("\nTest Distribution:\n", y_test.value_counts(normalize=True))


# ------------------------------------------
# 7. FEATURE SCALING
# ------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Complete")


# ------------------------------------------
# 8. FINAL OUTPUT READY
# ------------------------------------------
print("\n✅ Preprocessing complete!")
# AFTER SCALING

print("\nScaling Complete")


# ==========================================
# STEP 12 — HANDLE CLASS IMBALANCE
# ==========================================

from imblearn.over_sampling import SMOTE
import pandas as pd

print("\nBefore SMOTE:\n")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n")
print(pd.Series(y_train).value_counts())

print("\nNew Training Shape:", X_train.shape)