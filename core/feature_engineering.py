# ==========================================
# GEOLOGICAL HAZARD FEATURE ENGINEERING
# ==========================================

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("grid_dataset.csv")

print("Initial Shape:", df.shape)
print(df.head())


# ------------------------------------------
# 2. BASIC CLEANING
# ------------------------------------------

# Fix negative depths if any
df['eq_depth_mean'] = df['eq_depth_mean'].clip(lower=0)

# Fill NaNs (important for stability)
df['eq_count'] = df['eq_count'].fillna(0)
df['eq_mag_mean'] = df['eq_mag_mean'].fillna(0)
df['eq_depth_mean'] = df['eq_depth_mean'].fillna(0)


# ------------------------------------------
# 3. SEISMIC PRESENCE FEATURES
# ------------------------------------------

# Binary flag: does this grid have any earthquakes?
df['has_seismic_activity'] = (df['eq_count'] > 0).astype(int)

# Log-scaled frequency (handles skew)
df['eq_frequency_score'] = np.log1p(df['eq_count'])


# ------------------------------------------
# 4. PHYSICS-BASED FEATURES
# ------------------------------------------

# Seismic energy proxy (logarithmic nature of magnitude)
df['seismic_energy_proxy'] = 10 ** (1.5 * df['eq_mag_mean'])

# Interaction: magnitude vs depth (damage proxy)
df['mag_depth_ratio'] = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)


# ------------------------------------------
# 5. DEPTH CATEGORIZATION
# ------------------------------------------

def categorize_depth(d):
    if d == 0:
        return 0   # no data / no quake
    elif d < 70:
        return 1   # shallow
    elif d < 300:
        return 2   # intermediate
    else:
        return 3   # deep

df['depth_category'] = df['eq_depth_mean'].apply(categorize_depth)


# ------------------------------------------
# 6. GEOSPATIAL FEATURES
# ------------------------------------------

# Absolute latitude (tectonic relevance)
df['lat_abs'] = abs(df['lat_grid'])

# Seismic zone flag (Ring of Fire / subduction belt latitudes ≤ 60°)
df['lat_seismic_zone'] = (df['lat_abs'] <= 60).astype(int)

# Seismic intensity: combines energy magnitude with frequency
df['seismic_intensity'] = df['seismic_energy_proxy'] * df['eq_frequency_score']

# Weighted magnitude: surface damage proxy (high mag + shallow depth = dangerous)
df['weighted_mag'] = df['eq_mag_mean'] / (df['eq_depth_mean'] + 1)


# ------------------------------------------
# 7. SPATIAL SMOOTHING (CRITICAL FEATURE)
# ------------------------------------------

# Create grid pivot (lat x lon)
pivot = df.pivot_table(
    index='lat_grid',
    columns='lon_grid',
    values='eq_count',
    fill_value=0
)

# Apply Gaussian smoothing (captures neighborhood activity)
smoothed_values = gaussian_filter(pivot.values, sigma=1)

# Convert back to DataFrame
smoothed_df = pd.DataFrame(
    smoothed_values,
    index=pivot.index,
    columns=pivot.columns
)

# Flatten back to long format
smoothed_df = smoothed_df.stack().reset_index()
smoothed_df.columns = ['lat_grid', 'lon_grid', 'neighbor_eq_count']

# Merge back into main dataset
df = df.merge(smoothed_df, on=['lat_grid', 'lon_grid'], how='left')


# ------------------------------------------
# 8. OPTIONAL: NORMALIZE NEIGHBOR FEATURE
# ------------------------------------------

df['neighbor_eq_count'] = np.log1p(df['neighbor_eq_count'])


# ------------------------------------------
# 9. REMOVE REDUNDANT FEATURES
# ------------------------------------------

# Highly correlated feature (you already identified)
if 'eq_mag_max' in df.columns:
    df = df.drop(columns=['eq_mag_max'])


# ------------------------------------------
# 10. FINAL SANITY CHECK
# ------------------------------------------

print("\nFinal Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())


# ------------------------------------------
# 11. SAVE FINAL DATASET
# ------------------------------------------

df.to_csv("featured_dataset.csv", index=False)

print("\n✅ Feature engineering complete!")
print("Saved as: featured_dataset.csv")
