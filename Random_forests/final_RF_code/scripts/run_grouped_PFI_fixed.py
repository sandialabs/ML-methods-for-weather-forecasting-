import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys
from copy import deepcopy
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import REGIONS, HORIZONS, TUNING_DIR, PFI_DIR, REDUNDANT_COLS
from subseasonal.GPFI import (
    build_XY_data,
    grouped_pfi,
    get_clusters,
    load_train_test_data_dev,
    save_final_GPFI_dataset,
)
from subseasonal.tuning import fill_or_default, parse_maxdepth


def add_weeks_52(reference_dates, horizon):
    """Return YYYYWW forecast-valid dates for a fixed 52-week calendar."""
    ref = pd.to_numeric(pd.Index(reference_dates), errors="raise").to_numpy(dtype=int)

    year = ref // 100
    week = ref % 100
    offset = (week - 1) + int(horizon)

    forecast_year = year + (offset // 52)
    forecast_week = (offset % 52) + 1

    return (forecast_year * 100 + forecast_week).astype(int)


def build_dev_train_validation_xy(
    train_data,
    test_data,
    region,
    horizon,
    candidate_features,
):
    """
    Build training and validation examples without losing early validation weeks.

    Training examples are constructed strictly from the development training split,
    so no training target can cross into the validation period.

    Validation examples are constructed from a continuous train+validation series so
    that forecasts verifying at the beginning of the validation period may use lagged
    predictor history from the end of the training period. Rows are then retained only
    when their forecast-valid date belongs to the validation/test split.
    """
    # Keep model training exactly as before: targets must remain inside train_data.
    X_train, y_train = build_XY_data(
        train_data.copy(), region, horizon, candidate_features
    )

    # Use the continuous history only to construct validation predictors/targets.
    continuous_data = pd.concat([train_data, test_data], axis=0)
    continuous_data = continuous_data[~continuous_data.index.duplicated(keep="last")]
    continuous_data = continuous_data.sort_index()

    X_all, y_all = build_XY_data(
        continuous_data.copy(), region, horizon, candidate_features
    )

    forecast_dates = add_weeks_52(X_all.index, horizon)
    validation_dates = pd.to_numeric(
        pd.Index(test_data.index), errors="raise"
    ).to_numpy(dtype=int)

    validation_mask = np.isin(forecast_dates, validation_dates)

    X_test = X_all.loc[validation_mask].copy()
    y_test = y_all.loc[validation_mask].copy()
    selected_forecast_dates = forecast_dates[validation_mask]

    # Boundary/alignment checks: every validation target week should appear exactly once.
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

    return X_train, y_train, X_test, y_test


def main(mode="standard"):
    print(f"Running grouped PFI for mode {mode}")

    # Development split: train = 1980-2016, validation/test = 2017-2020.
    train_data, test_data = load_train_test_data_dev()

    # Load previously saved feature importances.
    # THIS IS ONLY FOR GETTING THE CORRECT FEATURES FOR EACH MODEL.
    importance_df = pd.read_csv(Path(PFI_DIR / "feature_importance.csv"))

    # Tuning files.
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

    # Train models for 1-4 week-ahead forecasting.
    for horizon in HORIZONS:
        print(f"Running group PFI for {horizon}-week ahead forecasting...")

        for region in REGIONS:
            print(f" - Processing region: {region}")

            row = tuning_df[
                (tuning_df["region"] == region)
                & (tuning_df["Horizon"] == horizon)
            ]

            candidate_features = importance_df[
                (importance_df["Region"] == region)
                & (importance_df["Horizon"] == horizon)
            ]["Feature"].tolist()

            # Boundary-safe construction: early 2017 validation forecasts may use
            # lagged predictor history from late 2016, while training targets remain
            # strictly inside the training period.
            X_train, y_train, X_test, y_test = build_dev_train_validation_xy(
                train_data,
                test_data,
                region,
                horizon,
                candidate_features,
            )

            # Train Random Forest Model.
            model = RandomForestRegressor(
                n_estimators=int(fill_or_default(row.iloc[0]["n_estimators"], 100)),
                max_depth=parse_maxdepth(row.iloc[0]["max_depth"]),
                max_features=fill_or_default(row.iloc[0]["max_features"], "sqrt"),
                min_samples_leaf=int(
                    fill_or_default(row.iloc[0]["min_samples_leaf"], 1)
                ),
                min_samples_split=int(
                    fill_or_default(row.iloc[0]["min_samples_split"], 2)
                ),
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            cluster_df = get_clusters(horizon, region)
            cluster_map = dict(zip(cluster_df["Feature"], cluster_df["Cluster"]))

            missing = [col for col in X_test.columns if col not in cluster_map]
            if missing:
                raise ValueError(f"Missing cluster assignments for: {missing[:10]}")

            group_dict = {}
            for col in X_test.columns:
                group_id = cluster_map[col]
                group_dict.setdefault(group_id, []).append(col)

            group_importance_df = grouped_pfi(
                model,
                X_test,
                y_test,
                group_dict,
                metric=mean_squared_error,
                n_repeats=20,
            )

            # Flatten the dictionary into a list of (group_id, variable) pairs.
            group_data = [
                (group_id, var)
                for group_id, vars in group_dict.items()
                for var in vars
            ]

            group_df = pd.DataFrame(group_data, columns=["Group", "Feature"])
            group_df = group_df.sort_values(by="Group").reset_index(drop=True)

            group_importance_df.to_csv(
                Path(PFI_DIR / mode / f"GPFI_{region}_{horizon}_{mode}.csv"),
                index=False,
            )
            group_df.to_csv(
                Path(PFI_DIR / mode / f"Groups_{region}_{horizon}_{mode}.csv"),
                index=False,
            )

    print("All PFI done, saving final file.")
    save_final_GPFI_dataset(mode=mode)
    print(f"Results saved to {PFI_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme", "weighted"],
        default="standard",
        help="Type of inference--extreme or all (standard) or weighted (weighted)",
    )
    args = parser.parse_args()
    main(args.mode)
