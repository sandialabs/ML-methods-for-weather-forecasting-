import pandas as pd
import numpy as np
import os
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

def process_RF_data(model="RF"):
    # climatology preds

    if model == "RF":
        preds = pd.read_csv(Path(PROJECT_ROOT / "outputs"/ "predictions_2021_2024_RF" / f"RF_preds_standard_2021_2024.csv"))
        # create forecast date
        year = preds["Reference_date"] // 100
        week = preds["Reference_date"] % 100
        
        offset = (week - 1) + preds["Horizon"]
        
        preds["Forecast_date"] = (
            (year + offset // 52) * 100
            + (offset % 52) + 1
        ).astype(int)

        RF_preds["RF_SE"] = (
        RF_preds["Pred_detrended"] -
        RF_preds["Obs_detrended"]
        )**2
        
        RF_preds["Clim_SE"] = (
            RF_preds["Obs_detrended"]
        )**2
        
        # positive = EESN better than climatology
        RF_preds["d_t"] = (
            RF_preds["Clim_SE"] -
            RF_preds["RF_SE"]
        )
        return RF_preds
    else:
        preds = pd.read_csv(Path(PROJECT_ROOT / "notebooks"/ "test_pred_obs.csv"))
        eesn["EESN_SE"] = (
        eesn["Pred_detrended"] -
        eesn["Obs_detrended"]
        )**2
        
        eesn["Clim_SE"] = (
            eesn["Obs_detrended"]
        )**2
        
        # positive = EESN better than climatology
        eesn["d_t"] = (
            eesn["Clim_SE"] -
            eesn["EESN_SE"]
        )
        return eesn


def block_bootstrap_mean_difference(
    x,
    block_length=8,
    n_boot=5000,
    seed=42
    ):
    """
    Circular block bootstrap for the mean paired loss difference.

    Parameters
    ----------
    x : array-like
        Time-ordered paired differences d_t.
        Positive values indicate RF outperforms climatology.
    block_length : int
        Number of consecutive weeks per resampled block.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    boot_means : np.ndarray
        Bootstrap distribution of mean(d_t).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    n = len(x)
    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        sampled_indices = []

        while len(sampled_indices) < n:
            start = rng.integers(0, n)

            block = (
                start + np.arange(block_length)
            ) % n

            sampled_indices.extend(block)

        sampled_indices = np.asarray(sampled_indices[:n])

        boot_means[b] = x[sampled_indices].mean()

    return boot_means

def main(model="RF"):
    print(f"Running bootstrap analysis on {model}")
    paired = process_RF_data(model=model)
    results = []
    
    for region in sorted(paired["Region"].unique()):
        for horizon in sorted(paired["Horizon"].unique()):
    
            sub = (
                paired[
                    (paired["Region"] == region) &
                    (paired["Horizon"] == horizon)
                ]
                .sort_values("Forecast_date")
                .dropna(subset=["d_t"])
            )
    
            d = sub["d_t"].to_numpy()
    
            boot = block_bootstrap_mean_difference(
                d,
                block_length=8,
                n_boot=5000,
                seed=42
            )
    
            mean_diff = d.mean()
            ci_low, ci_high = np.quantile(
                boot,
                [0.025, 0.975]
            )
    
            results.append({
                "Region": region,
                "Horizon": horizon,
                "N": len(d),
                "Mean_MSE_diff": mean_diff,
                "CI_lower": ci_low,
                "CI_upper": ci_high,
                "RF_better": ci_low > 0,
                "Climatology_better": ci_high < 0
            })
    
    bootstrap_results = pd.DataFrame(results)
    bootstrap_results.to_csv(Path(PROJECT_ROOT / "outputs"/ "predictions_2021_2024_RF" / f"bootstrap_results_{model}.csv")) 
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["RF", "EESN"],
        default="RF",
        help="Which model type to analyze",
    )
    args = parser.parse_args()
    main(args.model)
