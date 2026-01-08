# main.py
import sys
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Professional display and warning management
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure src directory is in the system path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.data_loader import (
    load_excel,
    clean_and_engineer_movies_df,
    clean_for_timeseries_lstm
)

from src.evaluation import (
    ensure_results_dirs,
    save_figure,
    save_csv,
    historical_dashboard_4plots,
    market_share_pies_2plots,
    market_share_evolution_4pies,
    roi_by_genre_bar,
    budget_vs_revenue_scatter_top10_distributors,
    correlation_heatmap,
    distributor_power_and_opening_2plots,
    seasonality_revenue_heatmap,
    genre_concentration_heatmap,
    release_volume_heatmap,
    intl_revenue_by_genre_bar,
    mean_budget_by_genre_bar,
    opening_boxplot_by_genre_log,
    opening_ratio_by_genre_bar,
    final_market_share_summary_4pies,
    kmeans_segmentation_scatter,
    kmeans_cluster_profiles_table,
    ml_comparison_barplot,
    ml_rf_feature_importance_plot,
    ml_lr_coefficients_plot,
    lstm_forecast_vs_actual_plot,
    lstm_training_loss_plot,
    plot_dynamic_seasonal_strategy,
    plot_historical_vs_prediction_validation,
    plot_opening_vs_total_pie,
    plot_lifecycle_forecast,
    diversified_strategic_matrix_plot
)

from src.models import (
    train_compare_models,
    prepare_lstm_data,
    train_lstm,
    strategic_forecast_simulator,
    generate_diversified_majors_plan
)

def main():
    # 0. Setup directories for results
    plots_dir, nums_dir = ensure_results_dirs("results")
    print("Initializing Movie Data Analytics Pipeline...")

    # 1. Data Ingestion
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "Data1.xlsx" # Looking for file in root
    
    if not excel_path.exists():
        excel_path = base_dir / "data" / "raw" / "Data1.xlsx"

    if not excel_path.exists():
        print(f"Error: Data1.xlsx not found at {excel_path}")
        return

    df_raw = load_excel(excel_path)
    df = clean_and_engineer_movies_df(df_raw)
    save_csv(df.head(2000), nums_dir, "00_preview_cleaned.csv")
    print("Data cleaning and feature engineering complete.")

    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis (Plots 01 to 15)
    # ---------------------------------------------------------
    print("--- Generating Historical & Market Analysis ---")
    save_figure(historical_dashboard_4plots(df), plots_dir, "01_historical_dashboard.png")
    
    fig, dist_pct, genre_pct = market_share_pies_2plots(df)
    save_figure(fig, plots_dir, "02_market_share_pies.png")
    save_csv(dist_pct, nums_dir, "02a_dist_share.csv")
    save_csv(genre_pct, nums_dir, "02b_genre_share.csv")

    save_figure(market_share_evolution_4pies(df), plots_dir, "03_market_share_evolution.png")
    
    fig, genre_roi = roi_by_genre_bar(df)
    save_figure(fig, plots_dir, "04_roi_by_genre.png")
    
    save_figure(budget_vs_revenue_scatter_top10_distributors(df), plots_dir, "05_budget_vs_revenue.png")
    
    fig, corr_matrix = correlation_heatmap(df)
    save_figure(fig, plots_dir, "06_correlation_heatmap.png")
    
    save_figure(distributor_power_and_opening_2plots(df), plots_dir, "07_distributor_power.png")
    save_figure(seasonality_revenue_heatmap(df)[0], plots_dir, "08_seasonality_heatmap.png")
    save_figure(genre_concentration_heatmap(df)[0], plots_dir, "09_genre_concentration.png")
    save_figure(release_volume_heatmap(df)[0], plots_dir, "10_release_volume.png")
    save_figure(intl_revenue_by_genre_bar(df), plots_dir, "11_intl_revenue_genre.png")
    save_figure(mean_budget_by_genre_bar(df), plots_dir, "12_mean_budget_genre.png")
    save_figure(opening_boxplot_by_genre_log(df), plots_dir, "13_opening_boxplot.png")
    save_figure(opening_ratio_by_genre_bar(df)[0], plots_dir, "14_opening_ratio.png")
    save_figure(final_market_share_summary_4pies(df), plots_dir, "15_final_market_summary.png")

    # ---------------------------------------------------------
    # 3. Clustering & Machine Learning (Plots 16 to 19)
    # ---------------------------------------------------------
    print("--- Executing Clustering & ML Models ---")
    fig, df_clust = kmeans_segmentation_scatter(df, n_clusters=4)
    save_figure(fig, plots_dir, "16_kmeans_segmentation.png")
    save_csv(kmeans_cluster_profiles_table(df_clust), nums_dir, "16_cluster_profiles.csv")

    comparison_df, metrics, artifacts = train_compare_models(df)
    save_csv(comparison_df, nums_dir, "17_ml_comparison_results.csv")
    save_figure(ml_comparison_barplot(comparison_df), plots_dir, "17_ml_comparison_plot.png")

    if artifacts.get("rf_importance_df") is not None:
        save_figure(ml_rf_feature_importance_plot(artifacts["rf_importance_df"]), plots_dir, "18_ml_feature_importance.png")
    if artifacts.get("lr_coef_df") is not None:
        save_figure(ml_lr_coefficients_plot(artifacts["lr_coef_df"]), plots_dir, "19_ml_lr_coefficients.png")

    # ---------------------------------------------------------
    # 4. Deep Learning & Monte Carlo Strategy (Plots 20 to 27)
    # ---------------------------------------------------------
    print("--- Starting LSTM Forecasting & Monte Carlo Risk Analysis ---")
    
    df_ts = clean_for_timeseries_lstm(df)
    X_seq, y_seq, scaler, split_info = prepare_lstm_data(df_ts, window_size=5)
    
    model_lstm, lstm_artifacts = train_lstm(
        X_seq, y_seq,
        train_size=split_info["train_size"],
        epochs=60,
        lr=0.01
    )

    # LSTM Diagnostics
    save_figure(lstm_forecast_vs_actual_plot(lstm_artifacts["y_test"], lstm_artifacts["preds"]), plots_dir, "20_lstm_forecast.png")
    save_figure(lstm_training_loss_plot(lstm_artifacts["losses"]), plots_dir, "21_lstm_loss.png")

    # Monte Carlo Strategic Simulation
    # Note: strategic_forecast_simulator now returns (best_rev, best_roi, mc_results_df)
    best_rev, best_roi, mc_results_df = strategic_forecast_simulator(df_ts, model_lstm, n_simulations=100)

    save_csv(mc_results_df, nums_dir, "22_monte_carlo_risk_results.csv")
    save_csv(pd.DataFrame([best_rev]), nums_dir, "22_best_revenue_scenario.csv")
    save_csv(pd.DataFrame([best_roi]), nums_dir, "22_best_roi_scenario.csv")

    # Strategy Visualizations using Monte Carlo data
    save_figure(plot_dynamic_seasonal_strategy(mc_results_df, best_rev, best_roi), plots_dir, "23_seasonal_strategy.png")
    save_figure(plot_historical_vs_prediction_validation(df_ts, mc_results_df), plots_dir, "24_historical_validation.png")
    save_figure(plot_opening_vs_total_pie(best_rev), plots_dir, "25_opening_pie.png")
    save_figure(plot_lifecycle_forecast(best_rev), plots_dir, "26_lifecycle_forecast.png")

    # Final Studio Matrix
    diversified_df = generate_diversified_majors_plan(model_lstm)
    save_csv(diversified_df, nums_dir, "27_diversified_majors_plan.csv")
    save_figure(diversified_strategic_matrix_plot(diversified_df), plots_dir, "27_diversified_studio_matrix.png")

    print("Done. All visual and numeric outputs generated in results/.")

if __name__ == "__main__":
    main()