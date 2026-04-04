"""
build_dataset.py
================
CAT II — Geological Hazard ML Classification
Rebuilds master_dataset.csv into a proper 1°×1° grid-cell dataset.

Inputs  : earthquakes.csv    — USGS earthquake data (160k rows, M4.0+)
          inform_clean.csv   — INFORM Risk Index (191 countries)

Output  : grid_dataset.csv   — ML-ready grid cell dataset

Usage   : python build_dataset.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("=" * 60)
print("  CAT II | Dataset Rebuild — 1°×1° Grid")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load & clean earthquakes.csv
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading earthquakes.csv ...")

eq = pd.read_csv("earthquakes.csv",
                 usecols=["time", "latitude", "longitude", "depth", "mag", "type"])

print(f"      Raw rows : {len(eq):,}")

# Keep only earthquake events (drops quarry blasts, explosions, etc.)
eq = eq[eq["type"] == "earthquake"].copy()

# Drop rows with missing lat, lon, mag or depth
eq = eq.dropna(subset=["latitude", "longitude", "mag", "depth"])

print(f"      After filter     : {len(eq):,} rows")
print(f"      Magnitude range  : {eq['mag'].min():.1f} - {eq['mag'].max():.1f}")
print(f"      Depth range      : {eq['depth'].min():.1f} - {eq['depth'].max():.1f} km")
print(f"      Date range       : {eq['time'].min()[:10]} to {eq['time'].max()[:10]}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — Snap each earthquake to 1°×1° grid cell centroid
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Snapping events to 1°×1° grid cells ...")

# Floor to get SW corner of cell, add 0.5 for centroid
eq["lat_grid"] = np.floor(eq["latitude"]).astype(int) + 0.5
eq["lon_grid"] = np.floor(eq["longitude"]).astype(int) + 0.5

unique_cells = eq[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]
print(f"      Unique cells with seismic activity : {unique_cells:,}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — Aggregate earthquake features per grid cell
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Aggregating features per cell ...")

eq_agg = eq.groupby(["lat_grid", "lon_grid"]).agg(
    eq_count      = ("mag",   "count"),
    eq_mag_mean   = ("mag",   "mean"),
    eq_mag_max    = ("mag",   "max"),
    eq_depth_mean = ("depth", "mean"),
).reset_index()

eq_agg["eq_mag_mean"]   = eq_agg["eq_mag_mean"].round(4)
eq_agg["eq_mag_max"]    = eq_agg["eq_mag_max"].round(4)
eq_agg["eq_depth_mean"] = eq_agg["eq_depth_mean"].round(4)

print(f"      Aggregated into {len(eq_agg):,} unique seismic grid cells")


# ─────────────────────────────────────────────────────────────
# STEP 4 — Build global land grid via GeoPandas spatial join
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Building global grid & matching to country polygons ...")

# Load Natural Earth country polygons directly from URL
# (geopandas.datasets was removed in GeoPandas 1.0)
NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)
print(f"      Downloading Natural Earth 110m polygons ...")
world = gpd.read_file(NE_URL)
world = world[["ISO_A3", "NAME", "CONTINENT", "geometry"]].rename(
    columns={"ISO_A3": "iso3", "NAME": "country_ne", "CONTINENT": "continent"}
)
world = world[world["iso3"] != "-99"].copy()

# Generate all 1°×1° cell centroids globally
lats = np.arange(-89.5, 90.5, 1.0)    # -89.5 ... 89.5
lons = np.arange(-179.5, 180.5, 1.0)  # -179.5 ... 179.5

lat_grid_vals, lon_grid_vals = np.meshgrid(lats, lons)
grid_df = pd.DataFrame({
    "lat_grid": lat_grid_vals.ravel(),
    "lon_grid": lon_grid_vals.ravel(),
})

print(f"      Total global cells generated : {len(grid_df):,}")

# Build GeoDataFrame with centroid points
grid_gdf = gpd.GeoDataFrame(
    grid_df,
    geometry=[Point(lon, lat)
              for lat, lon in zip(grid_df["lat_grid"], grid_df["lon_grid"])],
    crs="EPSG:4326",
)

# Spatial join — centroid point within country polygon
world_gdf = gpd.GeoDataFrame(world, geometry="geometry", crs="EPSG:4326")

joined = gpd.sjoin(
    grid_gdf,
    world_gdf[["iso3", "country_ne", "continent", "geometry"]],
    how="left",
    predicate="within",
)

# Keep only land cells matched to a country
joined = joined.drop_duplicates(subset=["lat_grid", "lon_grid"])
land   = joined[joined["iso3"].notna()].copy()
land   = land[["lat_grid", "lon_grid", "iso3", "country_ne", "continent"]].reset_index(drop=True)

print(f"      Land cells matched      : {len(land):,}")
print(f"      Ocean/unmatched dropped : {len(joined) - len(land):,}")


# ─────────────────────────────────────────────────────────────
# STEP 5 — Merge INFORM risk labels
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Merging INFORM risk labels ...")

inform = pd.read_csv("inform_clean.csv")
print(f"      INFORM countries loaded : {len(inform)}")

land = land.merge(
    inform[["iso3", "country", "inform_risk_score", "risk_label"]],
    on="iso3",
    how="left",
)

no_inform = land["risk_label"].isna().sum()
print(f"      Cells without INFORM match (dropped) : {no_inform:,}")
land = land[land["risk_label"].notna()].copy()
print(f"      Land cells with INFORM labels : {len(land):,}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — Merge earthquake features; fill zero-activity cells
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Merging seismic features ...")

final = land.merge(eq_agg, on=["lat_grid", "lon_grid"], how="left")

# Cells with no earthquake get zeros
final["eq_count"]      = final["eq_count"].fillna(0).astype(int)
final["eq_mag_mean"]   = final["eq_mag_mean"].fillna(0.0)
final["eq_mag_max"]    = final["eq_mag_max"].fillna(0.0)
final["eq_depth_mean"] = final["eq_depth_mean"].fillna(0.0)

# Final column order
final = final[[
    "country", "iso3", "continent",
    "lat_grid", "lon_grid",
    "inform_risk_score", "risk_label",
    "eq_count", "eq_mag_mean", "eq_mag_max", "eq_depth_mean",
]].reset_index(drop=True)

# Save
final.to_csv("grid_dataset.csv", index=False)


# ─────────────────────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────────────────────
active = (final["eq_count"] > 0).sum()

print("\n" + "=" * 60)
print("  BUILD COMPLETE")
print("=" * 60)
print(f"  Output file       : grid_dataset.csv")
print(f"  Total rows        : {len(final):,}")
print(f"  Total columns     : {len(final.columns)}")
print(f"  Countries covered : {final['iso3'].nunique()}")
print(f"  Continents        : {final['continent'].nunique()}")
print(f"  Active cells      : {active:,}  ({active/len(final)*100:.1f}%)")
print(f"  Zero-activity     : {len(final)-active:,}  ({(len(final)-active)/len(final)*100:.1f}%)")
print(f"\n  Risk label distribution:")
print(final["risk_label"].value_counts().reindex(
    ["Very High", "High", "Medium", "Low", "Very Low"]
).fillna(0).astype(int).to_string())
print(f"\n  Sample (active cells):")
print(final[final["eq_count"] > 0].head(5).to_string(index=False))
