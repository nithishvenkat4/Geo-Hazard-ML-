import pandas as pd

print("Cleaning INFORM Risk data...")
print("=" * 40)

# Load the sheet
df = pd.read_excel("inform_risk.xlsx", sheet_name="INFORM Risk Mid 2025 (a-z)")

# Row 0 has the real headers — promote it
df.columns = df.iloc[0]
df = df.drop(index=[0, 1])  # drop the header rows
df = df.reset_index(drop=True)

# Keep only the columns we need
df = df[['COUNTRY', 'ISO3', 'INFORM RISK']].copy()

# Rename for easier use
df.columns = ['country', 'iso3', 'inform_risk_score']

# Remove any empty rows
df = df.dropna(subset=['inform_risk_score'])

# Convert score to number
df['inform_risk_score'] = pd.to_numeric(df['inform_risk_score'], errors='coerce')
df = df.dropna(subset=['inform_risk_score'])

# Create Low / Medium / High label from the score
# INFORM scores go from 0 to 10
def assign_label(score):
    if score <= 3.5:
        return 'Low'
    elif score <= 6.0:
        return 'Medium'
    else:
        return 'High'

df['risk_label'] = df['inform_risk_score'].apply(assign_label)

# Save it
df.to_csv("inform_clean.csv", index=False)

print(f"Done! {len(df)} countries loaded")
print()
print("Risk label distribution:")
print(df['risk_label'].value_counts())
print()
print("Sample data:")
print(df.head(10))
