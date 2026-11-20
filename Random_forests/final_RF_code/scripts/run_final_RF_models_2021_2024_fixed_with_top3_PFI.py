import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import SPLITS, DETREND_DIR, HORIZONS, REGIONS, REGION_ZSCORES, PFI_DIR, PRED_DIR_FINAL, RES_DIR_FINAL, TUNING_DIR
from subseasonal.features import get_best_features_for_region_horizon, load_train_test_data_final, get_best_features_with_percent_tolerance
from subseasonal.io import read_csv_with_date
from subseasonal.tuning import fill_or_default, parse_maxdepth
from subseasonal.GPFI import build_XY_data
from subseasonal.metrics import weighted_rmse_step, weighted_rmse_smooth

def add_weeks_52(reference_dates, horizon):
    """
    Convert YYYYWW reference dates to forecast-valid YYYYWW dates using
    the manuscript's fixed 52-week calendar.
    """
    ref = pd.Series(pd.Index(reference_dates).astype(int), index=pd.Index(reference_dates))
    year = ref // 100
    week = ref % 100
    offset = (week - 1) + horizon
    return ((year + offset // 52) * 100 + (offset % 52) + 1).astype(int)


def get_percent_tolerance_selection_metadata(
    results_df,
    region,
    horizon,
    rmse_col="Test_RMSE",
    pct_tolerance=0.01,
):
    """
    Reproduce the exact percent-tolerance selection used for the final RF model,
    but also return the selected row and tolerance metadata.

    The iterative GPFI results are ordered from fewer to more groups, so taking
    the first row within the tolerance selects the smallest model whose score is
    within pct_tolerance of the minimum score.
    """
    sub_df = results_df[
        (results_df["Region"] == region)
        & (results_df["Horizon"] == horizon)
    ].copy()

    if sub_df.empty:
        raise ValueError(f"No iterative GPFI results for {region}, horizon {horizon}.")

    min_metric = sub_df[rmse_col].min()
    tolerance_threshold = min_metric * (1 + pct_tolerance)
    best_rows = sub_df[sub_df[rmse_col] <= tolerance_threshold]

    if best_rows.empty:
        raise ValueError(
            f"No model falls within {pct_tolerance:.1%} of the minimum "
            f"{rmse_col} for {region}, horizon {horizon}."
        )

    selected_row = best_rows.iloc[0].copy()
    return selected_row, min_metric, tolerance_threshold


def extract_top_selected_pfi_groups(
    gpfi_df,
    selected_features,
    selected_row,
    region,
    horizon,
    mode,
    rmse_col,
    pct_tolerance,
    min_metric,
    tolerance_threshold,
    top_n=3,
):
    """
    Return one row per feature for the top PFI groups that are actually part
    of the percent-tolerance-selected final model.

    Columns include group rank, original group ID, grouped permutation
    importance, its standard deviation, group size, and each feature belonging
    to the group. If the selected model contains fewer than `top_n` groups,
    only the groups that are actually in the final selected model are returned.
    """
    group_data = gpfi_df[
        (gpfi_df["Region"] == region)
        & (gpfi_df["Horizon"] == horizon)
        & (gpfi_df["Mean_Importance"] > 0)
    ].copy()

    if group_data.empty:
        raise ValueError(f"No positive grouped-PFI results for {region}, horizon {horizon}.")

    # One importance value is repeated for every feature in a group in the
    # merged GPFI file. Collapse to one row per group and rank exactly as the
    # iterative feature-selection script does.
    group_summary = (
        group_data.groupby("Group", as_index=False)
        .agg(
            Mean_Importance=("Mean_Importance", "mean"),
            Std_Importance=("Std_Importance", "mean"),
            Group_Size=("Group_Size", "first"),
        )
        .sort_values("Mean_Importance", ascending=False)
        .reset_index(drop=True)
    )

    selected_num_groups = int(selected_row["Num_Groups"])
    selected_group_summary = group_summary.iloc[:selected_num_groups].copy()

    # Sanity check: the union of features in the selected groups should be the
    # exact feature list saved by the iterative percent-tolerance selection.
    selected_group_ids = set(selected_group_summary["Group"])
    reconstructed_features = set(
        group_data.loc[group_data["Group"].isin(selected_group_ids), "Feature"].unique()
    )
    expected_features = set(selected_features)

    if reconstructed_features != expected_features:
        missing = sorted(expected_features - reconstructed_features)
        extra = sorted(reconstructed_features - expected_features)
        raise ValueError(
            f"PFI-group/feature mismatch for {region}, horizon {horizon}. "
            f"Missing from groups: {missing[:10]}; extra from groups: {extra[:10]}"
        )

    n_to_save = min(top_n, selected_num_groups)
    top_groups = selected_group_summary.iloc[:n_to_save].copy()
    top_groups["Group_Rank"] = np.arange(1, n_to_save + 1)

    # Long format: one row per feature, with group metadata repeated. This is
    # convenient for plotting and can also be collapsed into a table later.
    out = (
        group_data[["Group", "Feature"]]
        .drop_duplicates()
        .merge(top_groups, on="Group", how="inner", validate="many_to_one")
    )

    out["Region"] = region
    out["Horizon"] = horizon
    out["Mode"] = mode
    out["Selection_Metric"] = rmse_col
    out["Pct_Tolerance"] = pct_tolerance
    out["Minimum_Metric"] = min_metric
    out["Tolerance_Threshold"] = tolerance_threshold
    out["Selected_Model_Metric"] = selected_row[rmse_col]
    out["Selected_Num_Groups"] = selected_num_groups
    out["Selected_Num_Features"] = int(selected_row["Num_Features"])

    out = out[
        [
            "Region",
            "Horizon",
            "Mode",
            "Selection_Metric",
            "Pct_Tolerance",
            "Minimum_Metric",
            "Tolerance_Threshold",
            "Selected_Model_Metric",
            "Selected_Num_Groups",
            "Selected_Num_Features",
            "Group_Rank",
            "Group",
            "Mean_Importance",
            "Std_Importance",
            "Group_Size",
            "Feature",
        ]
    ].sort_values(["Region", "Horizon", "Group_Rank", "Feature"])

    return out

def main(mode="standard"):

    split_key = "final"
    spec = SPLITS[split_key] # this will always be final!
    train_end_year = spec["train_end_year"]
    train_data, test_data = load_train_test_data_final()

    # Keep the training construction strictly within the training period,
    # but retain the full chronological series for constructing TEST features.
    # This lets early-2021 forecasts use lagged values from late 2020 without
    # allowing any 2021 target to enter model fitting.
    full_data = pd.concat([train_data, test_data], axis=0).sort_index()
    test_end_year = test_data.index.str[:4].astype(int).max()

    weekly_aves = read_csv_with_date(spec["data_file"]).set_index("Date")
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > train_end_year] # is 2020

    # TS tuned parameters
    file_list = [Path(TUNING_DIR / mode / f"rf_tuning_results_MW_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_NE_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_SE_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_SW_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_W_tscv_{mode}.csv")]
  
    # preds from linear model on train set --  for backtransforming
    train_preds = read_csv_with_date(DETREND_DIR /  split_key / "train_preds.csv").set_index("Date")
    test_preds = read_csv_with_date(DETREND_DIR /  split_key / "test_preds.csv").set_index("Date")
    
    GPFI_results = pd.read_csv(PFI_DIR / f'iterative_group_rmse_results_RF_TS_{mode}.csv')

    # Merged grouped-PFI output: one row per feature, with group importance
    # repeated for all features in that group. Used below to export the top
    # groups belonging to each percent-tolerance-selected final RF model.
    merged_gpfi_df = pd.read_csv(PFI_DIR / f"All_GPFI_Merged_RF_TS_{mode}.csv")
    
    # Load previously saved feature importances
    importance_df = pd.read_csv(PFI_DIR / "feature_importance.csv")  
        
        
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
            
    tuning_df = pd.concat(df_list, axis=0)

    # Dictionary to store trained models and results
    rmse_results = []
    RFpreds = []  # List to accumulate all results
    top3_pfi_rows = []  # Long-format rows for reviewer feature-importance plots

        
    if mode == "standard":
        rmse_name = "Test_RMSE"
    elif mode == "weighted":
        rmse_name = "Weighted_RMSE_smooth"
    else:
        rmse_name = "Test_RMSE_extreme"
    
    # Train models for 1-4 week-ahead forecasting
    for horizon in HORIZONS:  # 1 to 4 weeks ahead
        print(f"Training models for {horizon}-week ahead forecasting...")
       
        for region, zscore_col in zip(REGIONS, REGION_ZSCORES):
            print(f" - Processing region: {region}")
            row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
            train_data = train_data.copy()
            test_data = test_data.copy()
    
            pct_tolerance = 0.01
            allowed_features, num_features = get_best_features_with_percent_tolerance(
                GPFI_results, region, horizon,
                rmse_col=rmse_name, pct_tolerance=pct_tolerance
            )

            # Retrieve the exact row selected by the same percent-tolerance
            # rule so we also know how many PFI groups comprise the final model.
            selected_row, min_metric, tolerance_threshold = (
                get_percent_tolerance_selection_metadata(
                    GPFI_results, region, horizon,
                    rmse_col=rmse_name, pct_tolerance=pct_tolerance
                )
            )

            if int(selected_row["Num_Features"]) != int(num_features):
                raise ValueError(
                    f"Feature-count mismatch for {region}, horizon {horizon}."
                )

            top3_pfi_rows.append(
                extract_top_selected_pfi_groups(
                    merged_gpfi_df,
                    selected_features=allowed_features,
                    selected_row=selected_row,
                    region=region,
                    horizon=horizon,
                    mode=mode,
                    rmse_col=rmse_name,
                    pct_tolerance=pct_tolerance,
                    min_metric=min_metric,
                    tolerance_threshold=tolerance_threshold,
                    top_n=3,
                )
            )
        
            
            # Training examples remain strictly within 1980-2020. Because
            # build_XY_data shifts the target forward by `horizon`, the final
            # `horizon` reference weeks of 2020 are automatically excluded from
            # training (their targets are outside the training split).
            X_train, y_train = build_XY_data(
                train_data.copy(), region, horizon, allowed_features
            )

            # Construct candidate evaluation examples on the CONTINUOUS series
            # so lagged predictors at the beginning of 2021 can come from 2020.
            # We then select rows by FORECAST-VALID date, not reference date.
            X_eval_all, y_eval_all = build_XY_data(
                full_data.copy(), region, horizon, allowed_features
            )
            forecast_dates_all = add_weeks_52(X_eval_all.index, horizon)
            forecast_years = forecast_dates_all // 100
            test_mask = (forecast_years > train_end_year) & (forecast_years <= test_end_year)

            X_test = X_eval_all.loc[test_mask].copy()
            y_test = y_eval_all.loc[test_mask].copy()
            forecast_dates = forecast_dates_all.loc[test_mask].copy()
            forecast_dates_str = forecast_dates.astype(str).tolist()

            # The climatological prediction, observed Kelvin temperature, and
            # Z-score are naturally indexed by the forecast-valid date.
            test_y_pred_subset = test_preds.loc[forecast_dates_str, region]
    
            # Train Random Forest Model
            model = RandomForestRegressor(
                    n_estimators=int(fill_or_default(row.iloc[0]['n_estimators'], 100)),
                    max_depth=parse_maxdepth(row.iloc[0]['max_depth']),
                    max_features=fill_or_default(row.iloc[0]['max_features'], 'sqrt'),
                    min_samples_leaf=int(fill_or_default(row.iloc[0]['min_samples_leaf'], 1)),
                    min_samples_split=int(fill_or_default(row.iloc[0]['min_samples_split'], 2)),
                    random_state=42,
                    n_jobs=-1
                )
            model.fit(X_train, y_train)
    
    
            # Predict on test set
            y_pred = model.predict(X_test)
    
            rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
            r_squared = r2_score(y_test, y_pred)
    
            # Backtransform residual forecasts using climatology at the
            # forecast-valid date.
            y_pred_kelvin = y_pred + test_y_pred_subset.to_numpy()
            y_test_kelvin = weekly_aves_test.loc[forecast_dates_str, region]
            z_data = weekly_aves_test.loc[forecast_dates_str, zscore_col]

            # Strong alignment check: the shifted residual target created above
            # must equal observed Kelvin temperature minus the climatological
            # prediction at the same forecast-valid date.
            residual_check = (
                y_test_kelvin.to_numpy() - test_y_pred_subset.to_numpy()
            )
            if not np.allclose(y_test.to_numpy(), residual_check, atol=1e-8, rtol=1e-8):
                raise ValueError(
                    f"Target/date alignment failed for {region}, horizon {horizon}."
                )

            # Extreme classification is also evaluated at the forecast-valid date.
            extreme_mask = ((z_data < -1) | (z_data > 1)).to_numpy()
            if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
                rmse_extreme = np.sqrt(mean_squared_error(y_test.to_numpy()[extreme_mask], y_pred[extreme_mask]))
            else:
                rmse_extreme = np.nan  # Assign NaN if no extremes exist

            # Convert to numpy arrays and remove any potential NaNs jointly
            valid_mask = (~pd.isna(y_test.values)) & (~pd.isna(y_pred)) & (~pd.isna(z_data.values))
            
            y_true_valid = y_test.values[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            z_valid = z_data.values[valid_mask]

            # compute weighted scores ala Taggart 2022 (this is NOT a proper score!)
            wrmse_step = weighted_rmse_step(y_true_valid, y_pred_valid, z_valid)
            wrmse_smooth = weighted_rmse_smooth(y_true_valid, y_pred_valid, z_valid, alpha=1.0)

            reference_dates = X_test.index.astype(int)

            # Create final dataframe for predictions
            predictions_df = pd.DataFrame({"Reference_date": reference_dates,
                                           "Forecast_date": forecast_dates.to_numpy(dtype=int),
                                           "Observed": y_test_kelvin.values,
                                           "Predicted": y_pred_kelvin,
                                           "Obs_detrended": y_test.values,
                                           "Pred_detrended": y_pred,
                                           "Horizon": horizon,
                                           "Region": region,
                                            "Z_score": z_data.values})
            RFpreds.append(predictions_df)
            
            rmse_results.append({
                "Region": region,
                "Horizon": horizon,
                "RMSE": rmse,
                "RMSE_extreme": rmse_extreme,
                "Weighted_RMSE_step": wrmse_step,
                "Weighted_RMSE_smooth": wrmse_smooth,
                "Num_features": num_features,
                "R_squared": r_squared, 
                "Feature_names": ";".join(sorted(allowed_features))
            })
    
            print(f"   - RMSE: {rmse:.4f}")
            print(f"   - RMSE extreme: {rmse_extreme:.4f}")
            print(f"   - R squared: {r_squared:.4f}")
            print(
                f"   - Evaluation dates: {forecast_dates.iloc[0]} to "
                f"{forecast_dates.iloc[-1]} (n={len(forecast_dates)})"
            )
    
    # Save RMSE results
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv(RES_DIR_FINAL / f"RF_rmse_results_{mode}_2021_2024.csv", index=False)
    
    final_preds = pd.concat(RFpreds, ignore_index=True)
    final_preds.to_csv(PRED_DIR_FINAL / f"RF_preds_{mode}_2021_2024.csv", index=False)

    # Save one row per feature for the top three grouped-PFI groups belonging
    # to each percent-tolerance-selected final model. This file is intended to
    # be easy to use directly for reviewer-requested feature-importance plots.
    top3_pfi_df = pd.concat(top3_pfi_rows, ignore_index=True)
    top3_pfi_file = PFI_DIR / f"RF_final_top3_grouped_PFI_{mode}.csv"
    top3_pfi_df.to_csv(top3_pfi_file, index=False)
    print(f"Top grouped-PFI information saved to {top3_pfi_file}")
    
    # Print completion message
    print("All models trained and predictions saved.")  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme", "weighted"],
        default="standard",
        help="Type of inference--extreme or all (standard)",
    )
    args = parser.parse_args()
    main(args.mode)