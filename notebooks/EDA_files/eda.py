import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

df = pd.read_csv("grid_dataset.csv")
df.loc[df['eq_depth_mean'] < 0, 'eq_depth_mean'] = 0.0
active = df[df['eq_count'] > 0].copy()

RISK_ORDER = ["Very High", "High", "Medium", "Low", "Very Low"]
PALETTE    = {"Very High":"#d62728","High":"#ff7f0e",
              "Medium":"#f0c030","Low":"#2ca02c","Very Low":"#1f77b4"}
df["risk_label"] = pd.Categorical(df["risk_label"], categories=RISK_ORDER, ordered=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,
                     "axes.spines.top":False,"axes.spines.right":False})

# ── FIG 1 — Risk Label Distribution ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13,5))
fig.suptitle("Fig 1 — INFORM Risk Label Distribution", fontweight="bold")
counts = df["risk_label"].value_counts().reindex(RISK_ORDER)
colors = [PALETTE[r] for r in RISK_ORDER]
axes[0].bar(RISK_ORDER, counts.values, color=colors, edgecolor="white")
axes[0].set_xlabel("Risk Category"); axes[0].set_ylabel("Grid Cells")
axes[0].set_title("Absolute Count")
for i,v in enumerate(counts.values):
    axes[0].text(i, v+50, f"{v:,}", ha="center", fontsize=9)
axes[1].pie(counts.values, labels=RISK_ORDER, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor":"white","linewidth":1.2})
axes[1].set_title("Proportional Split")
plt.tight_layout()
plt.savefig("eda_fig1_risk_distribution.png", bbox_inches="tight")
plt.close()
print("Fig 1 saved")

# ── FIG 2 — World Scatter Map ─────────────────────────────────
fig, ax = plt.subplots(figsize=(16,7))
ax.set_facecolor("#d4e9f7")
fig.patch.set_facecolor("#f7f9fc")
for risk in reversed(RISK_ORDER):
    sub = df[df["risk_label"]==risk]
    ax.scatter(sub["lon_grid"], sub["lat_grid"],
               c=PALETTE[risk], s=1.2, alpha=0.4, label=risk, linewidths=0)
sc = ax.scatter(active["lon_grid"], active["lat_grid"],
                c=active["eq_mag_mean"], cmap="hot_r",
                s=active["eq_count"].clip(1,100)*0.5,
                alpha=0.8, linewidths=0, vmin=4, vmax=8)
cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Mean Magnitude", fontsize=9)
ax.set_xlim(-180,180); ax.set_ylim(-90,90)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Fig 2 — Global Seismic Activity & Risk Level\n"
             "(dot size ∝ eq_count | colour ∝ mean magnitude)",
             fontweight="bold")
ax.legend(title="Risk Label", loc="lower left", markerscale=5,
          framealpha=0.8, fontsize=8)
plt.tight_layout()
plt.savefig("eda_fig2_world_map.png", bbox_inches="tight")
plt.close()
print("Fig 2 saved")

# ── FIG 3 — Correlation Heatmap ───────────────────────────────
feat = ["inform_risk_score","eq_count","eq_mag_mean",
        "eq_mag_max","eq_depth_mean"]
rename = {"inform_risk_score":"INFORM Score","eq_count":"EQ Count",
          "eq_mag_mean":"Mag Mean","eq_mag_max":"Mag Max",
          "eq_depth_mean":"Depth Mean"}
corr = df[feat].corr().rename(index=rename, columns=rename)
fig, ax = plt.subplots(figsize=(8,6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"shrink":0.8})
ax.set_title("Fig 3 — Feature Correlation Matrix", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("eda_fig3_correlations.png", bbox_inches="tight")
plt.close()
print("Fig 3 saved")

# ── FIG 4 — Magnitude & Depth by Risk (active cells only) ────
fig, axes = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Fig 4 — Seismic Features by Risk Label (active cells only)",
             fontweight="bold")
for ax, col, ylabel in zip(axes,
    ["eq_mag_mean","eq_depth_mean"],
    ["Mean Magnitude (Mw)","Mean Depth (km)"]):
    sns.boxplot(data=active, x="risk_label", y=col,
                order=RISK_ORDER, palette=PALETTE, ax=ax,
                linewidth=0.8, fliersize=2)
    ax.set_xlabel("Risk Label"); ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
plt.tight_layout()
plt.savefig("eda_fig4_mag_depth_boxplot.png", bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# ── FIG 5 — EQ Count by Continent ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Fig 5 — Earthquake Activity by Continent", fontweight="bold")
cont_eq = df.groupby("continent")["eq_count"].sum().sort_values(ascending=False)
axes[0].bar(cont_eq.index, cont_eq.values, color="#5b7fbe", edgecolor="white")
axes[0].set_xlabel("Continent"); axes[0].set_ylabel("Total EQ Count")
axes[0].set_title("Total Events per Continent")
axes[0].tick_params(axis='x', rotation=20)
for i,v in enumerate(cont_eq.values):
    axes[0].text(i, v+30, f"{v:,}", ha="center", fontsize=8)
cont_active = df[df["eq_count"]>0].groupby("continent").size()
cont_total  = df.groupby("continent").size()
pct = (cont_active / cont_total * 100).reindex(cont_eq.index)
axes[1].bar(pct.index, pct.values, color="#e07b3f", edgecolor="white")
axes[1].set_xlabel("Continent"); axes[1].set_ylabel("% Active Cells")
axes[1].set_title("% of Cells with Seismic Activity")
axes[1].tick_params(axis='x', rotation=20)
for i,v in enumerate(pct.values):
    axes[1].text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("eda_fig5_continent.png", bbox_inches="tight")
plt.close()
print("Fig 5 saved")

# ── FIG 6 — INFORM Score vs Seismic Activity ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Fig 6 — INFORM Risk Score vs Seismic Features", fontweight="bold")
axes[0].scatter(df["inform_risk_score"], df["eq_count"],
                c=[PALETTE[r] for r in df["risk_label"]],
                alpha=0.3, s=8, linewidths=0)
axes[0].set_xlabel("INFORM Risk Score"); axes[0].set_ylabel("EQ Count")
axes[0].set_title("Risk Score vs EQ Count")
axes[1].scatter(active["inform_risk_score"], active["eq_mag_mean"],
                c=[PALETTE[r] for r in active["risk_label"]],
                alpha=0.4, s=10, linewidths=0)
axes[1].set_xlabel("INFORM Risk Score"); axes[1].set_ylabel("Mean Magnitude")
axes[1].set_title("Risk Score vs Mean Magnitude (active only)")
from matplotlib.patches import Patch
handles = [Patch(color=PALETTE[r], label=r) for r in RISK_ORDER]
fig.legend(handles=handles, title="Risk Label",
           loc="lower center", ncol=5, bbox_to_anchor=(0.5,-0.04))
plt.tight_layout()
plt.savefig("eda_fig6_risk_vs_seismic.png", bbox_inches="tight")
plt.close()
print("Fig 6 saved")

# ── FIG 7 — Class Imbalance + Top 10 Countries ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Fig 7 — Class Imbalance & Top Seismic Countries", fontweight="bold")
counts2 = df["risk_label"].value_counts().reindex(RISK_ORDER)
bars = axes[0].bar(RISK_ORDER, counts2.values,
                   color=[PALETTE[r] for r in RISK_ORDER], edgecolor="white")
axes[0].set_xlabel("Risk Label"); axes[0].set_ylabel("Count")
axes[0].set_title("Class Imbalance Overview")
for bar, v in zip(bars, counts2.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f"{v:,}", ha="center", fontsize=9)
top10 = df.groupby("country")["eq_count"].sum().sort_values(ascending=False).head(10)
axes[1].barh(top10.index[::-1], top10.values[::-1], color="#5b7fbe", edgecolor="white")
axes[1].set_xlabel("Total EQ Count"); axes[1].set_title("Top 10 Countries by EQ Count")
plt.tight_layout()
plt.savefig("eda_fig7_imbalance_top10.png", bbox_inches="tight")
plt.close()
print("Fig 7 saved")

print("\nAll 7 EDA figures saved successfully.")
