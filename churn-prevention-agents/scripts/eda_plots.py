import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
df = pd.read_csv("/tmp/customer_churn_features.csv")

target = "churn_risk_score"
df["churn_label"] = df[target].map({1: "Churned", 0: "Retained"})

sns.set_theme(style="whitegrid", font_scale=1.1)
palette_churn = {"Churned": "#e74c3c", "Retained": "#2ecc71"}

# 1. Churn distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["churn_label"].value_counts()
bars = ax.bar(counts.index, counts.values, color=[palette_churn[x] for x in counts.index], edgecolor="white")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 300, f"{int(b.get_height()):,}", ha="center", fontweight="bold")
ax.set_title("Churn Distribution")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(OUT / "01_churn_distribution.png", dpi=150)
plt.close()

# 2. Churn rate by membership category
fig, ax = plt.subplots(figsize=(8, 4))
mem_churn = df.groupby("membership_category")[target].mean().sort_values(ascending=False)
colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in mem_churn.values]
mem_churn.plot.barh(ax=ax, color=colors, edgecolor="white")
ax.set_xlabel("Churn Rate")
ax.set_title("Churn Rate by Membership Category")
ax.axvline(0.5, ls="--", color="gray", alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "02_churn_by_membership.png", dpi=150)
plt.close()

# 3. Numeric feature distributions by churn
num_cols = ["age", "days_since_last_login", "avg_time_spent", "avg_transaction_value",
            "avg_frequency_login_days", "points_in_wallet", "tenure_days"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x="churn_label", y=col, ax=axes[i], palette=palette_churn, fliersize=1)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
if len(num_cols) < len(axes):
    axes[-1].set_visible(False)
fig.suptitle("Numeric Features by Churn Status", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "03_numeric_by_churn.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Correlation heatmap (numeric only)
num_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12, 10))
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap")
fig.tight_layout()
fig.savefig(OUT / "04_correlation_heatmap.png", dpi=150)
plt.close()

# 5. Churn by value_segment and engagement_segment
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, seg_col in enumerate(["value_segment", "engagement_segment"]):
    seg_rate = df.groupby(seg_col)[target].mean().sort_values(ascending=False)
    colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in seg_rate.values]
    seg_rate.plot.bar(ax=axes[i], color=colors, edgecolor="white")
    axes[i].set_title(f"Churn Rate by {seg_col}")
    axes[i].set_ylabel("Churn Rate")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
    axes[i].axhline(0.5, ls="--", color="gray", alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "05_churn_by_segments.png", dpi=150)
plt.close()

# 6. Complaint and support risk vs churn
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, col in enumerate(["past_complaint", "support_risk", "feedback"]):
    ct = pd.crosstab(df[col], df["churn_label"], normalize="index")
    ct[["Retained", "Churned"]].plot.bar(stacked=True, ax=axes[i],
        color=[palette_churn["Retained"], palette_churn["Churned"]], edgecolor="white")
    axes[i].set_title(f"Churn by {col}")
    axes[i].set_ylabel("Proportion")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
    axes[i].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "06_complaint_support_churn.png", dpi=150)
plt.close()

# 7. Tenure distribution by churn
fig, ax = plt.subplots(figsize=(8, 4))
for label, color in palette_churn.items():
    subset = df[df["churn_label"] == label]["tenure_days"]
    ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, edgecolor="white")
ax.set_xlabel("Tenure (days)")
ax.set_ylabel("Count")
ax.set_title("Tenure Distribution by Churn Status")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "07_tenure_distribution.png", dpi=150)
plt.close()

# 8. Price sensitivity vs churn
fig, ax = plt.subplots(figsize=(6, 4))
ps_rate = df.groupby("price_sensitivity")[target].mean().sort_values(ascending=False)
colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in ps_rate.values]
ps_rate.plot.bar(ax=ax, color=colors, edgecolor="white")
ax.set_title("Churn Rate by Price Sensitivity")
ax.set_ylabel("Churn Rate")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(0.5, ls="--", color="gray", alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "08_churn_by_price_sensitivity.png", dpi=150)
plt.close()

print("All 8 figures saved to:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f" - {f.name}")
