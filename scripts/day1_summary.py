import pandas as pd

df = pd.read_csv("master_dataset.csv")

print("=" * 40)
print("DAY 1 COMPLETE - Master Dataset")
print("=" * 40)
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print()
print("Features available:")
for col in df.columns:
    print(f"  - {col}")
print()
print("Label distribution:")
print(df['risk_label'].value_counts())
print()
print("Any missing values?", df.isnull().sum().sum())
print()
print("master_dataset.csv is ready for EDA tomorrow.")
