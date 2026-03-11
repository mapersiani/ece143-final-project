"""
Exploratory Data Analysis — plotting functions for the churn dataset.

Each function returns a matplotlib Figure so callers can display or save as needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font_scale=1.1)

PALETTE = {"Churned": "#e74c3c", "Retained": "#2ecc71"}
TARGET = "churn_risk_score"


def _label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["churn_label"] = out[TARGET].map({1: "Churned", 0: "Retained"})
    return out


def plot_churn_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of churned vs retained counts."""
    df = _label(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["churn_label"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[x] for x in counts.index], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 300,
                f"{int(b.get_height()):,}", ha="center", fontweight="bold")
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_churn_by_membership(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of churn rate per membership category."""
    fig, ax = plt.subplots(figsize=(8, 4))
    mem_churn = df.groupby("membership_category")[TARGET].mean().sort_values(ascending=False)
    colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in mem_churn.values]
    mem_churn.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Churn Rate")
    ax.set_title("Churn Rate by Membership Category")
    ax.axvline(0.5, ls="--", color="gray", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_numeric_by_churn(df: pd.DataFrame) -> plt.Figure:
    """Box plots of key numeric features split by churn status."""
    df = _label(df)
    num_cols = [
        "age", "days_since_last_login", "avg_time_spent", "avg_transaction_value",
        "avg_frequency_login_days", "points_in_wallet", "tenure_days",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x="churn_label", y=col, hue="churn_label",
                    ax=axes[i], palette=PALETTE, fliersize=1, legend=False)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    if len(num_cols) < len(axes):
        axes[-1].set_visible(False)
    fig.suptitle("Numeric Features by Churn Status", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Lower-triangle heatmap of numeric feature correlations."""
    num_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_churn_by_segments(df: pd.DataFrame) -> plt.Figure:
    """Side-by-side bar charts of churn rate by value_segment and engagement_segment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, seg_col in enumerate(["value_segment", "engagement_segment"]):
        seg_rate = df.groupby(seg_col)[TARGET].mean().sort_values(ascending=False)
        colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in seg_rate.values]
        seg_rate.plot.bar(ax=axes[i], color=colors, edgecolor="white")
        axes[i].set_title(f"Churn Rate by {seg_col}")
        axes[i].set_ylabel("Churn Rate")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
        axes[i].axhline(0.5, ls="--", color="gray", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_complaint_support_churn(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar charts of churn proportion by complaint, support risk, and feedback."""
    df = _label(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(["past_complaint", "support_risk", "feedback"]):
        ct = pd.crosstab(df[col], df["churn_label"], normalize="index")
        ct[["Retained", "Churned"]].plot.bar(
            stacked=True, ax=axes[i],
            color=[PALETTE["Retained"], PALETTE["Churned"]], edgecolor="white")
        axes[i].set_title(f"Churn by {col}")
        axes[i].set_ylabel("Proportion")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
        axes[i].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_tenure_distribution(df: pd.DataFrame) -> plt.Figure:
    """Overlapping histograms of tenure for churned vs retained."""
    df = _label(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in PALETTE.items():
        subset = df[df["churn_label"] == label]["tenure_days"]
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Tenure (days)")
    ax.set_ylabel("Count")
    ax.set_title("Tenure Distribution by Churn Status")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_churn_by_price_sensitivity(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of churn rate by price sensitivity level."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ps_rate = df.groupby("price_sensitivity")[TARGET].mean().sort_values(ascending=False)
    colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in ps_rate.values]
    ps_rate.plot.bar(ax=ax, color=colors, edgecolor="white")
    ax.set_title("Churn Rate by Price Sensitivity")
    ax.set_ylabel("Churn Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6)
    fig.tight_layout()
    return fig


def save_all_figures(df: pd.DataFrame, out_dir: str = "figures", dpi: int = 150) -> list[str]:
    """Generate and save all 8 EDA figures. Returns list of saved paths."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    plots = [
        ("01_churn_distribution.png", plot_churn_distribution),
        ("02_churn_by_membership.png", plot_churn_by_membership),
        ("03_numeric_by_churn.png", plot_numeric_by_churn),
        ("04_correlation_heatmap.png", plot_correlation_heatmap),
        ("05_churn_by_segments.png", plot_churn_by_segments),
        ("06_complaint_support_churn.png", plot_complaint_support_churn),
        ("07_tenure_distribution.png", plot_tenure_distribution),
        ("08_churn_by_price_sensitivity.png", plot_churn_by_price_sensitivity),
    ]

    saved = []
    for fname, func in plots:
        fig = func(df)
        path = out / fname
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved
