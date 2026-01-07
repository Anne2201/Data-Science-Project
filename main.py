# main.py
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import (
    load_excel,
    load_first_csv_in_dir,
    clean_and_engineer_movies_df,
    clean_for_timeseries_lstm
)

from src.evaluation import (
    ensure_results_dirs,
    save_figure,
    save_csv,

    # Exploratory data analysis (EDA) / historical
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

    # Clustering
    kmeans_segmentation_scatter,
    kmeans_cluster_profiles_table,

    # ML plots
    ml_comparison_barplot,
    ml_rf_feature_importance_plot,
    ml_lr_coefficients_plot,

    # LSTM plots and strategy
    lstm_forecast_vs_actual_plot,
    lstm_training_loss_plot,
    strategic_simulation_summary_table,  # numeric export
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
    plots_dir, nums_dir = ensure_results_dirs("results")

    # -------------------------
    # Data: Excel dataset (Data1.xlsx)
    # -------------------------
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "data" / "raw" / "Data1.xlsx"

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Not Found: {excel_path}\n"
            f"Put Data1.xlsx dans data/raw/Data1.xlsx"
        )

    df_raw = load_excel(excel_path)
    df = clean_and_engineer_movies_df(df_raw)

    # Save cleaned preview
    save_csv(df.head(2000), nums_dir, "00_preview_cleaned_head2000.csv")

    # =========================
    # 1) EDA PLOTS (≈ 20+)
    # =========================

    fig = historical_dashboard_4plots(df)
    save_figure(fig, plots_dir, "01_historical_dashboard_4plots.png")

    fig, dist_pct, genre_pct = market_share_pies_2plots(df)
    save_figure(fig, plots_dir, "02_market_share_pies_2plots.png")
    save_csv(dist_pct, nums_dir, "02a_distributor_market_share_pct.csv")
    save_csv(genre_pct, nums_dir, "02b_genre_market_share_pct.csv")

    fig = market_share_evolution_4pies(df)
    save_figure(fig, plots_dir, "03_market_share_evolution_4pies.png")

    fig, genre_roi = roi_by_genre_bar(df)
    save_figure(fig, plots_dir, "04_roi_by_genre_bar.png")
    save_csv(genre_roi, nums_dir, "04_roi_by_genre_median.csv")

    fig = budget_vs_revenue_scatter_top10_distributors(df)
    save_figure(fig, plots_dir, "05_budget_vs_revenue_top10_distributors.png")

    fig, corr_long = correlation_heatmap(df)
    save_figure(fig, plots_dir, "06_correlation_heatmap.png")
    save_csv(corr_long, nums_dir, "06_correlation_matrix_long.csv")

    fig = distributor_power_and_opening_2plots(df)
    save_figure(fig, plots_dir, "07_distributor_power_and_opening_2plots.png")

    fig, pivot_rev = seasonality_revenue_heatmap(df)
    save_figure(fig, plots_dir, "08_seasonality_revenue_heatmap.png")
    save_csv(pivot_rev.reset_index(), nums_dir, "08_seasonality_revenue_pivot.csv")

    fig, conc_tbl = genre_concentration_heatmap(df)
    save_figure(fig, plots_dir, "09_genre_concentration_heatmap.png")
    save_csv(conc_tbl.reset_index(), nums_dir, "09_genre_concentration_table.csv")

    fig, vol_tbl = release_volume_heatmap(df)
    save_figure(fig, plots_dir, "10_release_volume_heatmap.png")
    save_csv(vol_tbl.reset_index(), nums_dir, "10_release_volume_table.csv")

    fig = intl_revenue_by_genre_bar(df)
    save_figure(fig, plots_dir, "11_international_revenue_by_genre.png")

    fig = mean_budget_by_genre_bar(df)
    save_figure(fig, plots_dir, "12_mean_budget_by_genre.png")

    fig = opening_boxplot_by_genre_log(df)
    save_figure(fig, plots_dir, "13_opening_boxplot_by_genre_log.png")

    fig, ratio_tbl = opening_ratio_by_genre_bar(df)
    save_figure(fig, plots_dir, "14_opening_ratio_by_genre.png")
    save_csv(ratio_tbl, nums_dir, "14_opening_ratio_by_genre.csv")

    fig = final_market_share_summary_4pies(df)
    save_figure(fig, plots_dir, "15_final_market_share_summary_4pies.png")

    # =========================
    # 2) CLUSTERING
    # =========================
    fig, df_clust = kmeans_segmentation_scatter(df, n_clusters=4)
    save_figure(fig, plots_dir, "16_kmeans_segmentation_scatter.png")

    prof = kmeans_cluster_profiles_table(df_clust)
    save_csv(prof.reset_index(), nums_dir, "16_kmeans_cluster_profiles.csv")

    # =========================
    # 3) MACHINE LEARNING
    # =========================
    comparison_df, metrics, artifacts = train_compare_models(df)
    save_csv(comparison_df, nums_dir, "17_ml_model_comparison_r2.csv")
    save_csv(pd.DataFrame([metrics]), nums_dir, "17_ml_cv_metrics.csv")

    fig = ml_comparison_barplot(comparison_df)
    save_figure(fig, plots_dir, "17_ml_model_comparison_barplot.png")

    if artifacts.get("rf_importance_df") is not None:
        fig = ml_rf_feature_importance_plot(artifacts["rf_importance_df"])
        save_figure(fig, plots_dir, "18_ml_rf_feature_importance.png")
        save_csv(artifacts["rf_importance_df"], nums_dir, "18_ml_rf_feature_importance.csv")

    if artifacts.get("lr_coef_df") is not None:
        fig = ml_lr_coefficients_plot(artifacts["lr_coef_df"])
        save_figure(fig, plots_dir, "19_ml_lr_coefficients_top_bottom.png")
        save_csv(artifacts["lr_coef_df"], nums_dir, "19_ml_lr_coefficients.csv")

    # =========================
    # 4) LSTM and STRATEGY (Using Excel data)
    # =========================
    print("--- Démarrage de la partie LSTM (calculs en cours...) ---")
    
    # Use 'df' (Excel) instead of CSV 
    df_ts = clean_for_timeseries_lstm(df)

    X_seq, y_seq, scaler, split_info = prepare_lstm_data(df_ts, window_size=5)
    model, lstm_artifacts = train_lstm(
        X_seq, y_seq,
        train_size=split_info["train_size"],
        epochs=60,
        lr=0.01
    )

    # Forecast vs actual
    fig = lstm_forecast_vs_actual_plot(lstm_artifacts["y_test"], lstm_artifacts["preds"])
    save_figure(fig, plots_dir, "20_lstm_forecast_vs_actual.png")

    # Loss curve
    fig = lstm_training_loss_plot(lstm_artifacts["losses"])
    save_figure(fig, plots_dir, "21_lstm_training_loss.png")

    # Save numeric predictions
    out_pred = pd.DataFrame({
        "y_test_scaled": lstm_artifacts["y_test"],
        "pred_scaled": lstm_artifacts["preds"]
    })
    save_csv(out_pred, nums_dir, "20_lstm_predictions_scaled.csv")

    # Strategic simulation
    best_rev, best_roi, sim_df = strategic_forecast_simulator(df_ts, model)

    save_csv(sim_df, nums_dir, "22_strategic_simulation_results.csv")
    save_csv(pd.DataFrame([best_rev]), nums_dir, "22_best_rev_row.csv")
    save_csv(pd.DataFrame([best_roi]), nums_dir, "22_best_roi_row.csv")

    # Plot dynamic seasonal strategy
    fig = plot_dynamic_seasonal_strategy(sim_df, best_rev, best_roi)
    save_figure(fig, plots_dir, "23_dynamic_seasonal_strategy.png")

    # Validation historical vs prediction
    fig = plot_historical_vs_prediction_validation(df_ts, sim_df)
    save_figure(fig, plots_dir, "24_historical_vs_prediction_validation.png")

    # Opening vs total pie for best_rev
    fig = plot_opening_vs_total_pie(best_rev)
    save_figure(fig, plots_dir, "25_opening_vs_total_pie.png")

    # Lifecycle forecast
    fig = plot_lifecycle_forecast(best_rev)
    save_figure(fig, plots_dir, "26_lifecycle_forecast.png")

    # Diversified strategic matrix (FINAL GRAPH)
    diversified_df = generate_diversified_majors_plan(model)
    save_csv(diversified_df, nums_dir, "27_diversified_majors_plan.csv")

    fig = diversified_strategic_matrix_plot(diversified_df)
    save_figure(fig, plots_dir, "27_diversified_strategic_matrix_2026_2035.png")

    print(" Done. Tout est dans results/plots et results/numerics.")


if __name__ == "__main__":
    main()