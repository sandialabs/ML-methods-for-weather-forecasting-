import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy
from collections import defaultdict
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import (
    REGIONS,
    HORIZONS,
    TUNING_DIR,
    RAW_DIR,
    PFI_DIR,
    REDUNDANT_COLS,
    SPLITS,
    MAX_LAG,
)
from subseasonal.GPFI import load_train_test_data_dev
from subseasonal.io import read_csv_with_date
from subseasonal.tuning import fill_or_default, parse_maxdepth
from subseasonal.metrics import weighted_rmse_step, weighted_rmse_smooth


def add_weeks_52(reference_dates, horizon):
    """Return YYYYWW forecast-valid dates for a fixed 52-week calendar."""
    ref = pd.to_numeric(pd.Index(reference_dates), errors="raise").to_numpy(dtype=int)

    year = ref // 100
    week = ref % 100
    offset = (week - 1) + int(horizon)

    forecast_year = year + (offset // 52)
    forecast_week = (offset % 52) + 1

    return (forecast_year * 100 + forecast_week).astype(int)


def add_target_and_lag_columns(data, region, horizon):
    """Create the target, current-week predictor, and lagged residual predictors."""
    data = data.copy()

    data[f"{region}_target"] = data[region].shift(-horizon)
    data[f"{region}_current_week"] = data[region]

    for lag in range(1, MAX_LAG):
        data[f"{region}_lag_{lag}"] = data[region].shift(lag)

    return data


def train_and_evaluate_model_boundary_safe(
    train_data,
    test_data,
    tuning_df,
    selected_features,
    region,
    horizon,
    zscore_col,
    weekly_aves,
):
    """
    Train on the strict development-training split and evaluate on every validation
    target week (2017-2020), while allowing early validation forecasts to use lagged
    predictor history from the end of 2016.

    This intentionally preserves *exactly* selected_features. In particular, it does
    not automatically add every lag to a reduced feature subset.
    """
    row = tuning_df[
        (tuning_df["region"] == region)
        & (tuning_df["Horizon"] == horizon)
    ]

    target_col = f"{region}_target"

    # -----------------------------
    # Training data: strict split.
    # -----------------------------
    train_aug = add_target_and_lag_columns(train_data, region, horizon)
    train_subset = train_aug.dropna(subset=[target_col] + selected_features)

    X_train = train_subset[selected_features]
    y_train = train_subset[target_col]

    # ------------------------------------------------------------
    # Validation data: continuous history, then select by VALID date.
    # ------------------------------------------------------------
    continuous_data = pd.concat([train_data, test_data], axis=0)
    continuous_data = continuous_data[~continuous_data.index.duplicated(keep="last")]
    continuous_data = continuous_data.sort_index()
    eval_aug = add_target_and_lag_columns(continuous_data, region, horizon)

    eval_subset = eval_aug.dropna(subset=[target_col] + selected_features)

    forecast_dates = add_weeks_52(eval_subset.index, horizon)
    validation_dates = pd.to_numeric(
        pd.Index(test_data.index), errors="raise"
    ).to_numpy(dtype=int)

    validation_mask = np.isin(forecast_dates, validation_dates)
    test_subset = eval_subset.loc[validation_mask].copy()
    selected_forecast_dates = forecast_dates[validation_mask]

    # Check one-to-one coverage of the full validation target window.
    if len(np.unique(selected_forecast_dates)) != len(selected_forecast_dates):
        raise ValueError(
            f"Duplicate forecast-valid dates for {region}, horizon {horizon}."
        )

    missing_dates = set(validation_dates) - set(selected_forecast_dates)
    extra_dates = set(selected_forecast_dates) - set(validation_dates)
    if missing_dates or extra_dates:
        raise ValueError(
            f"Validation-date alignment failed for {region}, horizon {horizon}. "
            f"Missing: {sorted(missing_dates)[:10]}; "
            f"extra: {sorted(extra_dates)[:10]}"
        )

    X_test = test_subset[selected_features]
    y_test = test_subset[target_col]

    model = RandomForestRegressor(
        n_estimators=int(fill_or_default(row.iloc[0]["n_estimators"], 100)),
        max_depth=parse_maxdepth(row.iloc[0]["max_depth"]),
        max_features=fill_or_default(row.iloc[0]["max_features"], "sqrt"),
        min_samples_leaf=int(fill_or_default(row.iloc[0]["min_samples_leaf"], 1)),
        min_samples_split=int(fill_or_default(row.iloc[0]["min_samples_split"], 2)),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Align Z-scores directly to forecast-valid dates rather than shifting within
    # an already-hard-split validation frame. This also works for reference dates
    # that lie in late 2016 but verify in early 2017.
    forecast_date_labels = pd.Index(selected_forecast_dates.astype(str))
    missing_z_dates = forecast_date_labels.difference(weekly_aves.index)
    if len(missing_z_dates) > 0:
        raise ValueError(
            f"Missing Z-score dates for {region}, horizon {horizon}: "
            f"{missing_z_dates[:10].tolist()}"
        )

    z_values = weekly_aves.loc[forecast_date_labels, zscore_col].to_numpy()
    z_data = pd.Series(z_values, index=y_test.index, name=zscore_col)

    extreme_mask = (z_data < -1) | (z_data > 1)

    if extreme_mask.sum() > 0:
        rmse_extreme = np.sqrt(
            mean_squared_error(
                y_test.to_numpy()[extreme_mask.to_numpy()],
                y_pred[extreme_mask.to_numpy()],
            )
        )
    else:
        rmse_extreme = np.nan

    valid_mask = (
        (~pd.isna(y_test.to_numpy()))
        & (~pd.isna(y_pred))
        & (~pd.isna(z_data.to_numpy()))
    )

    y_true_valid = y_test.to_numpy()[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    z_valid = z_data.to_numpy()[valid_mask]

    wrmse_step = weighted_rmse_step(y_true_valid, y_pred_valid, z_valid)
    wrmse_smooth = weighted_rmse_smooth(
        y_true_valid, y_pred_valid, z_valid, alpha=1.0
    )

    return rmse, rmse_extreme, wrmse_step, wrmse_smooth


def main(mode="standard"):
    spec = SPLITS["dev"]  # this will always be dev!
    train_data, test_data = load_train_test_data_dev()

    # Keep the full 1980-2020 weekly dataframe for valid-date Z-score lookup.
    weekly_aves = read_csv_with_date(spec["data_file"]).set_index("Date")

    # TS tuned parameters.
    file_list = [
        Path(TUNING_DIR / mode / f"rf_tuning_results_MW_tscv_{mode}.csv"),
        Path(TUNING_DIR / mode / f"rf_tuning_results_NE_tscv_{mode}.csv"),
        Path(TUNING_DIR / mode / f"rf_tuning_results_SE_tscv_{mode}.csv"),
        Path(TUNING_DIR / mode / f"rf_tuning_results_SW_tscv_{mode}.csv"),
        Path(TUNING_DIR / mode / f"rf_tuning_results_W_tscv_{mode}.csv"),
    ]

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)

    tuning_df = pd.concat(df_list, axis=0)

    # Load grouped PFI results.
    gpfi_df = pd.read_csv(PFI_DIR / f"All_GPFI_Merged_RF_TS_{mode}.csv")

    # Filter to positive importance only.
    gpfi_df = gpfi_df[gpfi_df["Mean_Importance"] > 0]

    results = []

    # Loop over all region/horizon combinations.
    for (region, horizon), group_data in gpfi_df.groupby(["Region", "Horizon"]):
        print(f"Processing Region: {region}, Horizon: {horizon}", flush=True)

        group_means = (
            group_data.groupby("Group")
            .agg({"Mean_Importance": "mean"})
            .sort_values("Mean_Importance", ascending=False)
            .reset_index()
        )

        zscore_col = f"{region}_Zscore"

        for i in range(1, len(group_means) + 1):
            top_groups = group_means.iloc[:i]["Group"].values
            selected_features = group_data[
                group_data["Group"].isin(top_groups)
            ]["Feature"].unique()
            selected_features = list(selected_features)

            rmse, rmse_extreme, wrmse_step, wrmse_smooth = (
                train_and_evaluate_model_boundary_safe(
                    train_data,
                    test_data,
                    tuning_df,
                    selected_features,
                    region,
                    horizon,
                    zscore_col,
                    weekly_aves,
                )
            )

            results.append(
                {
                    "Region": region,
                    "Horizon": horizon,
                    "Num_Groups": i,
                    "Num_Features": len(selected_features),
                    "Test_RMSE": rmse,
                    "Test_RMSE_extreme": rmse_extreme,
                    "Weighted_RMSE_step": wrmse_step,
                    "Weighted_RMSE_smooth": wrmse_smooth,
                    "Feature_names": ";".join(sorted(selected_features)),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        PFI_DIR / f"iterative_group_rmse_results_RF_TS_{mode}_New.csv",
        index=False,
    )

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme", "weighted"],
        default="standard",
        help="run standard (standard) or only on extremes (extreme) or weighted (weighted)",
    )
    args = parser.parse_args()
    main(args.mode)
