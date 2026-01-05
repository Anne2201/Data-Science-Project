# src/evaluation.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def ensure_results_dirs(root: str = "results") -> Tuple[Path, Path]:
    root = Path(root)
    plots_dir = root / "plots"
    nums_dir = root / "numerics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    nums_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, nums_dir


def save_figure(fig: plt.Figure, plots_dir: Path, filename: str, dpi: int = 170) -> None:
    out = plots_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_csv(df: pd.DataFrame, nums_dir: Path, filename: str) -> None:
    df.to_csv(nums_dir / filename, index=False)


def _safe_filter_year(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    d = df.copy()
    if "Year" in d.columns:
        d = d.dropna(subset=["Year"])
        d = d[(d["Year"] >= y0) & (d["Year"] <= y1)]
    return d


# -------------------------
# EDA / HISTORICAL (Notebook-style)
# -------------------------
def historical_dashboard_4plots(df: pd.DataFrame) -> plt.Figure:
    d = _safe_filter_year(df, 1937, 2023)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Historical Evolution of the Movie Industry (1937–2023)", fontsize=18)

    # a) volume
    if "Year" in d.columns:
        count = d.groupby("Year").size()
        axes[0, 0].plot(count.index, count.values, marker="o", linewidth=1.6)
    axes[0, 0].set_title("Evolution of Movie Production Volume")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Number of Movies")
    axes[0, 0].grid(True, alpha=0.25)

    # b) budget mean
    if {"Year", "Budget (in $)"}.issubset(d.columns):
        b = d.groupby("Year")["Budget (in $)"].mean()
        axes[0, 1].plot(b.index, b.values, linewidth=1.6)
    axes[0, 1].set_title("Average Budget Growth ($)")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Mean Budget")
    axes[0, 1].grid(True, alpha=0.25)

    # c) revenue WW vs International
    if {"Year", "World Wide Sales (in $)", "International Sales (in $)"}.issubset(d.columns):
        rev = d.groupby("Year")[["World Wide Sales (in $)", "International Sales (in $)"]].mean()
        axes[1, 0].plot(rev.index, rev["World Wide Sales (in $)"], label="World Wide", linewidth=1.6)
        axes[1, 0].plot(rev.index, rev["International Sales (in $)"], label="International", linewidth=1.6)
        axes[1, 0].legend()
    axes[1, 0].set_title("Average Revenue: World Wide vs International")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Revenue ($)")
    axes[1, 0].grid(True, alpha=0.25)

    # d) all genres evolution (top 10 to keep readable)
    if {"Year", "Main_Genre"}.issubset(d.columns):
        g = d[d["Main_Genre"] != "Unknown"].groupby(["Year", "Main_Genre"]).size().unstack(fill_value=0)
        top_cols = g.sum().sort_values(ascending=False).head(10).index
        for col in top_cols:
            axes[1, 1].plot(g.index, g[col], linewidth=1.2, alpha=0.85, label=str(col))
        axes[1, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].set_title("Production Trends: Genres Evolution (Top 10)")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Number of Movies")
    axes[1, 1].grid(True, alpha=0.25)

    return fig


def market_share_pies_2plots(df: pd.DataFrame):
    d = _safe_filter_year(df, 1990, 2023)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Market Share Analysis (Revenue Based: 1990–2023)", fontsize=16)

    # distributor
    dist_sales = d.groupby("Distributor")["World Wide Sales (in $)"].sum().sort_values(ascending=False)
    top = dist_sales.head(8)
    others = pd.Series({"Others": dist_sales.iloc[8:].sum()})
    final_dist = pd.concat([top, others])
    axes[0].pie(final_dist.values, labels=final_dist.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[0].set_title("Market Share by Distributor (Top 8 + Others)")
    dist_pct = (final_dist / final_dist.sum() * 100).round(2).reset_index()
    dist_pct.columns = ["Distributor", "MarketSharePct"]

    # genre
    genre_sales = d.groupby("Main_Genre")["World Wide Sales (in $)"].sum().sort_values(ascending=False)
    topg = genre_sales.head(8)
    othersg = pd.Series({"Others": genre_sales.iloc[8:].sum()})
    final_gen = pd.concat([topg, othersg])
    axes[1].pie(final_gen.values, labels=final_gen.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Market Share by Main Genre (Top 8 + Others)")
    genre_pct = (final_gen / final_gen.sum() * 100).round(2).reset_index()
    genre_pct.columns = ["Main_Genre", "MarketSharePct"]

    return fig, dist_pct, genre_pct


def market_share_evolution_4pies(df: pd.DataFrame) -> plt.Figure:
    era_1 = _safe_filter_year(df, 1937, 1970)
    era_2 = _safe_filter_year(df, 2000, 2023)

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle("Evolution of Industry Structure: Classic vs Modern Era", fontsize=18)

    def top6_plus_others(data, col):
        s = data.groupby(col)["World Wide Sales (in $)"].sum().sort_values(ascending=False)
        top = s.head(6)
        others = pd.Series({"Others": s.iloc[6:].sum()})
        return pd.concat([top, others])

    d1 = top6_plus_others(era_1, "Distributor")
    d2 = top6_plus_others(era_2, "Distributor")
    g1 = top6_plus_others(era_1, "Main_Genre")
    g2 = top6_plus_others(era_2, "Main_Genre")

    axes[0, 0].pie(d1.values, labels=d1.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[0, 0].set_title("Distributor Market Share (1937–1970)")

    axes[0, 1].pie(d2.values, labels=d2.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[0, 1].set_title("Distributor Market Share (2000–2023)")

    axes[1, 0].pie(g1.values, labels=g1.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[1, 0].set_title("Genre Market Share (1937–1970)")

    axes[1, 1].pie(g2.values, labels=g2.index.astype(str), autopct="%1.1f%%", startangle=140)
    axes[1, 1].set_title("Genre Market Share (2000–2023)")

    return fig


def roi_by_genre_bar(df: pd.DataFrame):
    d = _safe_filter_year(df, 1990, 2023).copy()
    d["ROI"] = np.where(d["Budget (in $)"] > 0, d["World Wide Sales (in $)"] / d["Budget (in $)"], np.nan)
    genre_roi = d.groupby("Main_Genre")["ROI"].median().sort_values(ascending=False)

    fig = plt.figure(figsize=(16, 8))
    plt.bar(genre_roi.index.astype(str), genre_roi.values)
    plt.axhline(1, linestyle="--", linewidth=2)
    plt.title("Profitability Analysis: Median ROI by Genre (1990–2023)")
    plt.ylabel("Median ROI (Revenue/Budget)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)

    return fig, genre_roi.reset_index(name="MedianROI")


def budget_vs_revenue_scatter_top10_distributors(df: pd.DataFrame) -> plt.Figure:
    d = df.copy()
    top_distribs = d.groupby("Distributor")["World Wide Sales (in $)"].sum().sort_values(ascending=False).head(10).index
    dt = d[d["Distributor"].isin(top_distribs)].copy()

    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for name, sub in dt.groupby("Distributor"):
        ax.scatter(sub["Budget (in $)"], sub["World Wide Sales (in $)"], alpha=0.6, label=str(name), s=30)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Budgetary Strategy: Budget vs Revenue (Top 10 Distributors)")
    ax.set_xlabel("Budget (log $)")
    ax.set_ylabel("World Wide Sales (log $)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    return fig


def correlation_heatmap(df: pd.DataFrame):
    d = _safe_filter_year(df, 1990, 2023).copy()
    feats = ["Budget (in $)", "World Wide Sales (in $)", "Domestic Opening (in $)", "Running Time", "ROI", "International Sales (in $)"]
    feats = [c for c in feats if c in d.columns]

    # top 6 genres as binary
    if "Main_Genre" in d.columns:
        top = d["Main_Genre"].value_counts().head(6).index
        for g in top:
            col = f"Genre_{g}"
            d[col] = (d["Main_Genre"] == g).astype(int)
            feats.append(col)

    corr = d[feats].corr()

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap: Success Factors (1990–2023)")
    plt.colorbar()

    corr_long = corr.stack().reset_index()
    corr_long.columns = ["Feature1", "Feature2", "Correlation"]
    return fig, corr_long


def distributor_power_and_opening_2plots(df: pd.DataFrame) -> plt.Figure:
    d = _safe_filter_year(df, 1990, 2023).copy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Distributor Power and Opening Performance (1990–2023)", fontsize=16)

    dist_total = d.groupby("Distributor")["World Wide Sales (in $)"].sum().sort_values(ascending=False).head(10)
    axes[0].barh(dist_total.index.astype(str), dist_total.values)
    axes[0].set_title("Top 10 Distributors by Total Revenue")
    axes[0].grid(True, axis="x", alpha=0.25)

    if "Domestic Opening (in $)" in d.columns:
        genre_open = d.groupby("Main_Genre")["Domestic Opening (in $)"].median().sort_values(ascending=False).head(12)
        axes[1].barh(genre_open.index.astype(str), genre_open.values)
        axes[1].set_title("Median Domestic Opening by Genre")
        axes[1].grid(True, axis="x", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "Domestic Opening missing", ha="center", va="center")

    return fig


def seasonality_revenue_heatmap(df: pd.DataFrame):
    d = _safe_filter_year(df, 1990, 2023).copy()
    top_genres = d["Main_Genre"].value_counts().head(10).index
    d = d[d["Main_Genre"].isin(top_genres)]

    pivot = d.pivot_table(index="Main_Genre", columns="Release_Month", values="World Wide Sales (in $)", aggfunc="mean").fillna(0)

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str))
    plt.title("Seasonality Strategy: Average Revenue by Genre & Release Month")
    plt.colorbar()

    return fig, pivot


def genre_concentration_heatmap(df: pd.DataFrame):
    d = df.copy()
    genre_cols = [c for c in d.columns if c.startswith("is_")]
    if not genre_cols:
        top = d["Main_Genre"].value_counts().head(7).index
        for g in top:
            d[f"is_{g}"] = (d["Main_Genre"] == g).astype(int)
        genre_cols = [c for c in d.columns if c.startswith("is_")]

    conc = d.groupby("Release_Month")[genre_cols].mean()
    if 0 in conc.index:
        conc = conc.drop(index=0)

    fig = plt.figure(figsize=(14, 7))
    plt.imshow(conc.values, aspect="auto")
    plt.yticks(range(len(conc.index)), conc.index.astype(str))
    plt.xticks(range(len(conc.columns)), [c.replace("is_", "") for c in conc.columns], rotation=45, ha="right")
    plt.title("Market Specialization: Genre Concentration by Release Month")
    plt.colorbar()

    return fig, conc


def release_volume_heatmap(df: pd.DataFrame):
    d = df.copy()
    pivot = d.pivot_table(index="Release_Month", columns="Main_Genre", values="Title", aggfunc="count").fillna(0)
    if 0 in pivot.index:
        pivot = pivot.drop(index=0)

    fig = plt.figure(figsize=(14, 8))
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.title("Market Saturation: Number of Releases by Month and Genre")
    plt.colorbar()

    return fig, pivot


def intl_revenue_by_genre_bar(df: pd.DataFrame) -> plt.Figure:
    d = df.copy()
    if "International Sales (in $)" not in d.columns:
        d["International Sales (in $)"] = d["World Wide Sales (in $)"] * 0.4

    intl = d.groupby("Main_Genre")["International Sales (in $)"].mean().sort_values(ascending=False)

    fig = plt.figure(figsize=(12, 6))
    plt.bar(intl.index.astype(str), intl.values)
    plt.title("Average International Revenue by Genre")
    plt.ylabel("Mean International Sales ($)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)
    return fig


def mean_budget_by_genre_bar(df: pd.DataFrame) -> plt.Figure:
    d = df.copy()
    bud = d.groupby("Main_Genre")["Budget (in $)"].mean().sort_values(ascending=False)

    fig = plt.figure(figsize=(12, 6))
    plt.bar(bud.index.astype(str), bud.values)
    plt.title("Mean Production Budget by Genre")
    plt.ylabel("Mean Budget ($)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)
    return fig


def opening_boxplot_by_genre_log(df: pd.DataFrame) -> plt.Figure:
    d = df.copy()
    d = d.dropna(subset=["Domestic Opening (in $)", "Main_Genre"])
    # keep reasonable top genres
    top = d["Main_Genre"].value_counts().head(10).index
    d = d[d["Main_Genre"].isin(top)]

    groups = [d.loc[d["Main_Genre"] == g, "Domestic Opening (in $)"].values for g in top]

    fig = plt.figure(figsize=(14, 7))
    plt.boxplot(groups, labels=[str(g) for g in top], showfliers=False)
    plt.yscale("log")
    plt.title("Distribution of Domestic Opening by Genre (Log Scale)")
    plt.ylabel("Domestic Opening ($) - Log Scale")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)
    return fig


def opening_ratio_by_genre_bar(df: pd.DataFrame):
    d = df.copy()
    d = d.dropna(subset=["Domestic Opening (in $)", "World Wide Sales (in $)", "Main_Genre"])
    d["Opening_Ratio"] = (d["Domestic Opening (in $)"] / d["World Wide Sales (in $)"]) * 100
    ratio = d.groupby("Main_Genre")["Opening_Ratio"].mean().sort_values(ascending=False)

    fig = plt.figure(figsize=(12, 6))
    plt.bar(ratio.index.astype(str), ratio.values)
    plt.title("Opening Weekend Dependency: % of Total Revenue (Mean)")
    plt.ylabel("Opening Ratio (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)

    ratio_tbl = ratio.reset_index()
    ratio_tbl.columns = ["Main_Genre", "Opening_Ratio_MeanPct"]
    return fig, ratio_tbl


def final_market_share_summary_4pies(df: pd.DataFrame) -> plt.Figure:
    d = _safe_filter_year(df, 1990, 2023).copy()
    d["Decade"] = (d["Year"] // 10) * 10

    d["Runtime_Bin"] = pd.cut(
        d["Running Time"],
        bins=[0, 90, 120, 150, 500],
        labels=["Short (<90m)", "Standard (90-120m)", "Long (120-150m)", "Epic (>150m)"]
    )

    d["Budget_Bin"] = pd.cut(
        d["Budget (in $)"],
        bins=[0, 20e6, 50e6, 100e6, 1e12],
        labels=["Low (<20M)", "Medium (20-50M)", "High (50-100M)", "Blockbuster (>100M)"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Final Synthesis: Global Market Share Breakdown (1990–2023)", fontsize=18)

    def pie(ax, s, title):
        s = s.dropna()
        ax.pie(s.values, labels=s.index.astype(str), autopct="%1.1f%%", startangle=140)
        ax.set_title(title)

    pie(axes[0, 0], d.groupby("Runtime_Bin")["World Wide Sales (in $)"].sum(), "Revenue by Movie Duration")
    pie(axes[0, 1], d.groupby("Budget_Bin")["World Wide Sales (in $)"].sum(), "Revenue by Budget Level")

    genre_sales = d.groupby("Main_Genre")["World Wide Sales (in $)"].sum().sort_values(ascending=False)
    top = genre_sales.head(6)
    others = pd.Series({"Others": genre_sales.iloc[6:].sum()})
    pie(axes[1, 0], pd.concat([top, others]), "Revenue by Film Type (Top 6 + Others)")

    decade_sales = d.groupby(d["Decade"].astype(str) + "s")["World Wide Sales (in $)"].sum()
    pie(axes[1, 1], decade_sales, "Revenue Weight by Decade")

    return fig


# -------------------------
# CLUSTERING
# -------------------------
def kmeans_segmentation_scatter(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    d = _safe_filter_year(df, 1990, 2023).copy()
    X = d[["Budget (in $)", "World Wide Sales (in $)"]].copy()
    X = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    d["Cluster"] = km.fit_predict(X_scaled)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for cl, sub in d.groupby("Cluster"):
        ax.scatter(sub["Budget (in $)"], sub["World Wide Sales (in $)"], label=f"Cluster {cl}", alpha=0.65, s=35)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Movie Market Segmentation: Budget vs Sales (Log Scale)")
    ax.set_xlabel("Budget (log $)")
    ax.set_ylabel("World Wide Sales (log $)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    return fig, d


def kmeans_cluster_profiles_table(df_clust: pd.DataFrame) -> pd.DataFrame:
    prof = df_clust.groupby("Cluster").agg({
        "Budget (in $)": "mean",
        "World Wide Sales (in $)": "mean",
        "ROI": "mean"
    }).round(2)
    return prof


# -------------------------
# ML plots
# -------------------------
def ml_comparison_barplot(comparison_df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(12, 7))
    plt.barh(comparison_df["Model"].astype(str), comparison_df["R2"].values)
    plt.title("Benchmark Summary: R2 by Model")
    plt.xlabel("R2")
    plt.grid(True, axis="x", alpha=0.25)
    return fig


def ml_rf_feature_importance_plot(importance_df: pd.DataFrame) -> plt.Figure:
    top = importance_df.sort_values("Importance", ascending=False).head(15)
    fig = plt.figure(figsize=(12, 7))
    plt.barh(top["Feature"].astype(str), top["Importance"].values)
    plt.title("Random Forest Feature Importance (Top 15)")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.25)
    return fig


def ml_lr_coefficients_plot(coef_df: pd.DataFrame) -> plt.Figure:
    top = pd.concat([coef_df.head(12), coef_df.tail(12)])
    fig = plt.figure(figsize=(12, 8))
    plt.barh(top["Feature"].astype(str), top["Coefficient"].values)
    plt.title("Linear Regression Coefficients (Top +/-)")
    plt.axvline(0, linewidth=1)
    plt.grid(True, axis="x", alpha=0.25)
    return fig


# -------------------------
# LSTM + STRATEGY plots
# -------------------------
def lstm_forecast_vs_actual_plot(y_test, preds) -> plt.Figure:
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    ax.plot(y_test, label="Actual (Scaled)", linewidth=2, alpha=0.7)
    ax.plot(preds, label="LSTM Forecast (Scaled)", linestyle="--", linewidth=2)
    ax.set_title("LSTM Forecast: Revenue Trajectory (Test Set)")
    ax.set_xlabel("Time index (test)")
    ax.set_ylabel("Scaled Log Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def lstm_training_loss_plot(losses) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    return fig


def strategic_simulation_summary_table(best_rev: Dict[str, Any], best_roi: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Type": "Max Revenue", **best_rev}, {"Type": "Max ROI", **best_roi}])


def plot_dynamic_seasonal_strategy(sim_df: pd.DataFrame, best_rev: Dict[str, Any], best_roi: Dict[str, Any]) -> plt.Figure:
    top_genres = sorted(list(set([best_rev["Genre"], best_roi["Genre"]])))
    plot_data = sim_df[sim_df["Genre"].isin(top_genres)].copy()

    fig = plt.figure(figsize=(14, 7))
    ax = plt.gca()

    for g in top_genres:
        sub = plot_data[plot_data["Genre"] == g].groupby("Month")["Forecasted_Revenue_M"].mean().reset_index()
        ax.plot(sub["Month"], sub["Forecasted_Revenue_M"], marker="o", linewidth=2, label=g)

    ax.set_title("Strategic Release Windows: Monthly Revenue Potential by Top Genres")
    ax.set_xlabel("Release Month")
    ax.set_ylabel("Projected Worldwide Sales (Millions USD)")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def plot_historical_vs_prediction_validation(df_ts: pd.DataFrame, sim_df: pd.DataFrame) -> plt.Figure:
    # historical = mean Revenue_M by month (from Log_Sales if exists)
    d = df_ts.copy()
    if "Log_Sales" in d.columns:
        d["Revenue_M"] = np.expm1(d["Log_Sales"]) / 1e6
    else:
        d["Revenue_M"] = 0.0

    hist = d.groupby("Release_Month")["Revenue_M"].mean().reset_index()
    pred = sim_df.groupby("Month")["Forecasted_Revenue_M"].mean().reset_index()

    # align
    merged = pd.merge(hist, pred, left_on="Release_Month", right_on="Month", how="inner").sort_values("Month")

    fig = plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.plot(hist["Release_Month"], hist["Revenue_M"], linestyle="--", marker="s", linewidth=2, alpha=0.7, label="Historical Market Average")
    ax.plot(pred["Month"], pred["Forecasted_Revenue_M"], marker="o", linewidth=3, label="LSTM Strategic Forecast")

    if not merged.empty:
        ax.fill_between(merged["Month"], merged["Revenue_M"], merged["Forecasted_Revenue_M"], alpha=0.2, label="Optimization Gain")

    ax.set_title("Validation: Historical Reality vs Optimized LSTM Strategy")
    ax.set_xlabel("Release Month")
    ax.set_ylabel("Average Revenue (Millions USD)")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.legend(loc="upper left")
    return fig


def plot_opening_vs_total_pie(best_design: Dict[str, Any]) -> plt.Figure:
    total_rev = float(best_design.get("Forecasted_Revenue_M", best_design.get("Forecasted_Rev_M", 0.0)))
    opening = total_rev * 0.35
    remaining = total_rev * 0.65

    fig = plt.figure(figsize=(9, 9))
    plt.pie(
        [opening, remaining],
        labels=["Opening Weekend (35%)", "Remaining Run (65%)"],
        autopct="%1.1f%%",
        startangle=140,
        explode=(0.08, 0.0),
        shadow=True
    )
    plt.title(f"Strategic Revenue Distribution\nTotal Forecast: ${total_rev:.1f}M")
    return fig


def plot_lifecycle_forecast(best_design: Dict[str, Any]) -> plt.Figure:
    total_revenue_m = float(best_design.get("Forecasted_Revenue_M", best_design.get("Forecasted_Rev_M", 0.0)))
    budget_m = float(best_design.get("Budget_M", 0.0))

    weeks = [0, 1, 2, 4, 8, 12]
    traj = [0, total_revenue_m * 0.35, total_revenue_m * 0.60, total_revenue_m * 0.85, total_revenue_m * 0.95, total_revenue_m]

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(weeks, traj, marker="o", linewidth=3, label="Predicted Cumulative Revenue")
    ax.fill_between(weeks, traj, alpha=0.12)
    ax.axhline(budget_m, linestyle="--", linewidth=2, label=f"Production Budget (${budget_m}M)")
    ax.set_title("Revenue Lifecycle Forecast (12 weeks)")
    ax.set_xlabel("Weeks Since Release")
    ax.set_ylabel("Cumulative Worldwide Sales (Millions USD)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    return fig


def diversified_strategic_matrix_plot(diversified_df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()

    # bubble size = forecasted rev
    for _, row in diversified_df.iterrows():
        ax.scatter(
            row["Budget_M"],
            row["Projected_ROI"],
            s=max(50, row["Forecasted_Rev_M"] * 5),
            alpha=0.7
        )
        ax.text(
            row["Budget_M"],
            row["Projected_ROI"] + 0.08,
            f"{row['Studio']}\n({row['Strategic_Genre']})",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title("DIVERSIFIED STRATEGIC MATRIX: Studio Specialization (2026–2035)")
    ax.set_xlabel("Budget Investment (Millions $)")
    ax.set_ylabel("Projected ROI (x)")
    ax.grid(True, alpha=0.2)
    return fig