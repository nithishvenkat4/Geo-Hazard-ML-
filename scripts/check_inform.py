import pandas as pd

print("Loading INFORM Risk sheet...")
print("=" * 40)

df = pd.read_excel("inform_risk.xlsx", sheet_name="INFORM Risk Mid 2025 (a-z)")

print("Shape:", df.shape)
print()
print("Column names:")
for col in df.columns:
    print(" -", col)
print()
print("First 3 rows:")
print(df.head(3))
