# scripts/run_qrf_region_horizon_calibration.py

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn_quantile import RandomForestQuantileRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import (
    SPLITS,
    DETREND_DIR,
    HORIZONS,
    REGIONS,
    PFI_DIR,
    TUNING_DIR,
    CONFORMAL_DIR
)
from subseasonal.features import (
    load_train_test_data_dev,
    load_train_test_data_final,
    get_best_features_with_percent_tolerance,
)
from subseasonal.GPFI import build_XY_data
from subseasonal.io import read_csv_with_date
from subseasonal.metrics import weighted_rmse_step, weighted_rmse_smooth


Q_LEVELS = [0.05, 0.50, 0.95]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_qrf_prediction(pred, q_levels=Q_LEVELS):
    """
    sklearn_quantile output shape can be either:
      - n_quantiles x n_samples
      - n_samples x n_quantiles

    This returns a DataFrame with columns q05, q50, q95.
    """
    arr = np.asarray(pred)

    n_q = len(q_levels)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D QRF prediction array, got shape {arr.shape}")

    if arr.shape[0] == n_q:
        arr = arr.T
    elif arr.shape[1] == n_q:
        pass
    else:
        raise ValueError(
            f"Could not infer QRF prediction orientation. "
            f"Shape={arr.shape}, expected one dimension to equal {n_q}."
        )

    col_map = {
        0.05: "q05",
        0.50: "q50",
        0.95: "q95",
    }

    cols = [col_map[q] for q in q_levels]
    return pd.DataFrame(arr, columns=cols)


def get_reference_and_forecast_dates(reference_index, full_date_index, horizon):
    """
    For each reference date t, forecast_date is t + horizon weekly steps,
    using the date ordering in the full dataset.
    """
    date_series = pd.Series(full_date_index, index=full_date_index)
    forecast_dates = date_series.shift(-horizon).loc[reference_index]
    return pd.Series(reference_index, index=reference_index), forecast_dates


def get_target_kelvin_and_trend(
    *,
    weekly_aves,
    trend_preds,
    region,
    horizon,
    reference_index,
):
    """
    y in build_XY_data is detrended residual target.
    We back-transform using the linear trend prediction at the target week.

    y_true_kelvin(t+h) = weekly_aves[region].shift(-horizon)
    trend_pred_K(t+h) = trend_preds[region].shift(-horizon)
    """

    y_true_kelvin = weekly_aves[region].shift(-horizon).loc[reference_index]
    trend_pred_K = trend_preds[region].shift(-horizon).loc[reference_index]

    return y_true_kelvin, trend_pred_K


def find_zscore_target(
    *,
    weekly_aves,
    region,
    horizon,
    reference_index,
):
    """
    Try to recover target-week z-score if a matching column exists.
    If no z-score column exists, return NaN.
    """


    col = f"{region}_Zscore"
    if col in weekly_aves.columns:
        return weekly_aves[col].shift(-horizon).loc[reference_index]

    return pd.Series(np.nan, index=reference_index)


def make_prediction_dataframe(
    *,
    X,
    y_resid,
    q_pred_resid,
    weekly_aves,
    trend_preds,
    region,
    horizon,
    dataset_label,
    include_features=True,
):
    """
    Build a standardized output dataframe for either calibration or test.
    """

    reference_index = y_resid.index

    reference_dates, forecast_dates = get_reference_and_forecast_dates(
        reference_index=reference_index,
        full_date_index=weekly_aves.index,
        horizon=horizon,
    )

    y_true_kelvin, trend_pred_K = get_target_kelvin_and_trend(
        weekly_aves=weekly_aves,
        trend_preds=trend_preds,
        region=region,
        horizon=horizon,
        reference_index=reference_index,
    )

    zscore_true = find_zscore_target(
        weekly_aves=weekly_aves,
        region=region,
        horizon=horizon,
        reference_index=reference_index,
    )

    q_pred_resid = q_pred_resid.reset_index(drop=True)

    out = pd.DataFrame({
        "dataset": dataset_label,
        "reference_date": reference_dates.values,
        "forecast_date": forecast_dates.values,
        "Region": region,
        "horizon": horizon,

        # detrended target and QRF predictions
        "y": y_resid.values,
        "q05_resid": q_pred_resid["q05"].values,
        "q50_resid": q_pred_resid["q50"].values,
        "q95_resid": q_pred_resid["q95"].values,

        # trend and Kelvin back-transform
        "trend_pred_K": trend_pred_K.values,
        "y_true_kelvin": y_true_kelvin.values,
        "q05_kelvin": q_pred_resid["q05"].values + trend_pred_K.values,
        "q50_kelvin": q_pred_resid["q50"].values + trend_pred_K.values,
        "q95_kelvin": q_pred_resid["q95"].values + trend_pred_K.values,

        # aliases for compatibility with earlier CP code
        "y_pred_resid": q_pred_resid["q50"].values,
        "y_pred_kelvin": q_pred_resid["q50"].values + trend_pred_K.values,

        "zscore_true": zscore_true.values,
    })

    out["qrf_width_resid"] = out["q95_resid"] - out["q05_resid"]
    out["qrf_width_kelvin"] = out["q95_kelvin"] - out["q05_kelvin"]

    out["qrf_covered"] = (
        (out["y_true_kelvin"] >= out["q05_kelvin"])
        & (out["y_true_kelvin"] <= out["q95_kelvin"])
    )

    if include_features:
        X_out = X.reset_index(drop=True).copy()
        X_out.columns = [f"feature__{c}" for c in X_out.columns]
        out = pd.concat([out.reset_index(drop=True), X_out], axis=1)

    return out


def compute_prediction_metrics(df):
    """
    Compute simple metrics using q50 as the point prediction.
    """

    valid = df[["y", "q50_resid"]].dropna()

    if len(valid) == 0:
        return {
            "RMSE_resid": np.nan,
            "RMSE_resid_extreme": np.nan,
            "Weighted_RMSE_step": np.nan,
            "Weighted_RMSE_smooth": np.nan,
            "N": 0,
            "N_extreme": 0,
            "QRF_coverage": np.nan,
            "Median_QRF_width_kelvin": np.nan,
        }

    rmse_all = rmse(valid["y"], valid["q50_resid"])

    extreme_mask = df["zscore_true"].abs() > 1 if "zscore_true" in df.columns else pd.Series(False, index=df.index)

    if extreme_mask.any():
        rmse_ext = rmse(
            df.loc[extreme_mask, "y"],
            df.loc[extreme_mask, "q50_resid"],
        )
    else:
        rmse_ext = np.nan

    if "zscore_true" in df.columns and df["zscore_true"].notna().any():
        valid_w = df[["y", "q50_resid", "zscore_true"]].dropna()
        wrmse_step = weighted_rmse_step(
            valid_w["y"].to_numpy(),
            valid_w["q50_resid"].to_numpy(),
            valid_w["zscore_true"].to_numpy(),
        )
        wrmse_smooth = weighted_rmse_smooth(
            valid_w["y"].to_numpy(),
            valid_w["q50_resid"].to_numpy(),
            valid_w["zscore_true"].to_numpy(),
            alpha=1.0,
        )
    else:
        wrmse_step = np.nan
        wrmse_smooth = np.nan

    return {
        "RMSE_resid": rmse_all,
        "RMSE_resid_extreme": rmse_ext,
        "Weighted_RMSE_step": wrmse_step,
        "Weighted_RMSE_smooth": wrmse_smooth,
        "N": int(len(df)),
        "N_extreme": int(extreme_mask.sum()) if "zscore_true" in df.columns else 0,
        "QRF_coverage": float(df["qrf_covered"].mean()),
        "Median_QRF_width_kelvin": float(df["qrf_width_kelvin"].median()),
    }


def run_qrf_region_horizon(
    *,
    mode="standard",
    pct_tolerance=0.01,
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=2,
    save_models=False,
    include_features=True,
):
    """
    Train one QRF per Region x Horizon using dev train data,
    save calibration predictions on dev test data,
    and save final test predictions on final test data.

    Outputs:
      - 20 calibration files
      - 20 test files
      - metrics summary CSV
      - optional fitted QRF models
    """

    out_root = ensure_dir(Path(CONFORMAL_DIR) / "qrf_region_horizon")
    calib_dir = ensure_dir(out_root / "calibration")
    test_dir = ensure_dir(out_root / "test")
    model_dir = ensure_dir(out_root / "models")

    # -----------------------------
    # Load dev split: train 1980-2016, calibration 2017-2020
    # -----------------------------
    dev_key = "dev"
    dev_spec = SPLITS[dev_key]
    train_data_dev, calib_data_dev = load_train_test_data_dev()

    weekly_aves_dev = read_csv_with_date(dev_spec["data_file"]).set_index("Date")
    trend_preds_calib = read_csv_with_date(
        DETREND_DIR / dev_key / "test_preds.csv"
    ).set_index("Date")

    # -----------------------------
    # Load final split: test 2021-2024
    # -----------------------------
    final_key = "final"
    final_spec = SPLITS[final_key]
    _, test_data_final = load_train_test_data_final()

    weekly_aves_final = read_csv_with_date(final_spec["data_file"]).set_index("Date")
    trend_preds_test = read_csv_with_date(
        DETREND_DIR / final_key / "test_preds.csv"
    ).set_index("Date")

    # -----------------------------
    # Feature selection results
    # -----------------------------
    gpfi_results = pd.read_csv(
        PFI_DIR / "iterative_group_rmse_results_RF_TS_standard.csv"
    )

    if mode == "standard":
        rmse_name = "Test_RMSE"
    elif mode == "weighted":
        rmse_name = "Weighted_RMSE_smooth"
    elif mode == "extreme":
        rmse_name = "Test_RMSE_extreme"
    else:
        raise ValueError("mode must be one of: standard, weighted, extreme")

    metrics_rows = []
    models_by_h_r = {}

    for horizon in HORIZONS:
        print(f"\n================ HORIZON {horizon} ================")

        for region in REGIONS:
            print(f"Training QRF | Region={region} | Horizon={horizon}")

            allowed_features, num_features = get_best_features_with_percent_tolerance(
                gpfi_results,
                region,
                horizon,
                rmse_col=rmse_name,
                pct_tolerance=pct_tolerance,
            )

            # -----------------------------
            # Build train/calibration matrices
            # -----------------------------
            X_train, y_train = build_XY_data(
                train_data_dev,
                region,
                horizon,
                allowed_features,
            )

            X_calib, y_calib = build_XY_data(
                calib_data_dev,
                region,
                horizon,
                allowed_features,
            )

            # Drop rows with missing values
            train_valid = pd.concat([X_train, y_train.rename("y")], axis=1).dropna()
            X_train = train_valid[X_train.columns]
            y_train = train_valid["y"]

            calib_valid = pd.concat([X_calib, y_calib.rename("y")], axis=1).dropna()
            X_calib = calib_valid[X_calib.columns]
            y_calib = calib_valid["y"]

            # -----------------------------
            # Fit QRF model
            # -----------------------------
            qrf_model = RandomForestQuantileRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                q=Q_LEVELS,
                random_state=42,
                n_jobs=-1,
            )

            qrf_model.fit(X_train, y_train)
            models_by_h_r[(region, horizon)] = qrf_model

            if save_models:
                model_path = model_dir / f"qrf_{region}_H{horizon}.joblib"
                joblib.dump(
                    {
                        "model": qrf_model,
                        "allowed_features": list(allowed_features),
                        "region": region,
                        "horizon": horizon,
                        "q_levels": Q_LEVELS,
                    },
                    model_path,
                )

            # -----------------------------
            # Predict calibration period: 2017-2020
            # -----------------------------
            pred_calib_raw = qrf_model.predict(X_calib)
            pred_calib = normalize_qrf_prediction(pred_calib_raw, q_levels=Q_LEVELS)

            calib_df = make_prediction_dataframe(
                X=X_calib,
                y_resid=y_calib,
                q_pred_resid=pred_calib,
                weekly_aves=weekly_aves_dev,
                trend_preds=trend_preds_calib,
                region=region,
                horizon=horizon,
                dataset_label="calibration",
                include_features=include_features,
            )

            calib_file = calib_dir / f"QRF_calibration_{region}_H{horizon}.csv"
            calib_df.to_csv(calib_file, index=False)

            calib_metrics = compute_prediction_metrics(calib_df)
            metrics_rows.append({
                "dataset": "calibration",
                "Region": region,
                "horizon": horizon,
                "Num_features": num_features,
                "Feature_names": ";".join(sorted(map(str, allowed_features))),
                **calib_metrics,
            })

            # -----------------------------
            # Build and predict final test period: 2021-2024
            # -----------------------------
            X_test, y_test = build_XY_data(
                test_data_final,
                region,
                horizon,
                allowed_features,
            )

            test_valid = pd.concat([X_test, y_test.rename("y")], axis=1).dropna()
            X_test = test_valid[X_test.columns]
            y_test = test_valid["y"]

            pred_test_raw = qrf_model.predict(X_test)
            pred_test = normalize_qrf_prediction(pred_test_raw, q_levels=Q_LEVELS)

            test_df = make_prediction_dataframe(
                X=X_test,
                y_resid=y_test,
                q_pred_resid=pred_test,
                weekly_aves=weekly_aves_final,
                trend_preds=trend_preds_test,
                region=region,
                horizon=horizon,
                dataset_label="test",
                include_features=include_features,
            )

            test_file = test_dir / f"QRF_test_{region}_H{horizon}.csv"
            test_df.to_csv(test_file, index=False)

            test_metrics = compute_prediction_metrics(test_df)
            metrics_rows.append({
                "dataset": "test",
                "Region": region,
                "horizon": horizon,
                "Num_features": num_features,
                "Feature_names": ";".join(sorted(map(str, allowed_features))),
                **test_metrics,
            })

            print(
                f"  Calibration: coverage={calib_metrics['QRF_coverage']:.3f}, "
                f"RMSE={calib_metrics['RMSE_resid']:.3f}, "
                f"median width={calib_metrics['Median_QRF_width_kelvin']:.3f}"
            )
            print(
                f"  Test:        coverage={test_metrics['QRF_coverage']:.3f}, "
                f"RMSE={test_metrics['RMSE_resid']:.3f}, "
                f"median width={test_metrics['Median_QRF_width_kelvin']:.3f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_root / "qrf_region_horizon_metrics.csv", index=False)

    print("\nDone.")
    print(f"Calibration files saved to: {calib_dir}")
    print(f"Test files saved to:        {test_dir}")
    print(f"Metrics saved to:           {out_root / 'qrf_region_horizon_metrics.csv'}")

    return metrics_df, models_by_h_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme", "weighted"],
        default="standard",
        help="Feature-selection criterion to use.",
    )
    parser.add_argument(
        "--pct_tolerance",
        type=float,
        default=0.01,
        help="Percent tolerance for grouped feature selection.",
    )
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument(
        "--no_save_models",
        action="store_true",
        help="Do not save fitted QRF models.",
    )
    parser.add_argument(
        "--no_features",
        action="store_true",
        help="Do not include feature columns in output prediction files.",
    )

    args = parser.parse_args()

    run_qrf_region_horizon(
        mode=args.mode,
        pct_tolerance=args.pct_tolerance,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        save_models=not args.no_save_models,
        include_features=not args.no_features,
    )


if __name__ == "__main__":
    main()