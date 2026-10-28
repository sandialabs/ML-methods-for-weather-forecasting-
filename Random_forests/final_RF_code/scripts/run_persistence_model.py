from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import REGIONS, REGION_ZSCORES, PERSISTENCE_DIR, SPLITS, HORIZONS
from subseasonal.io import read_csv_with_date
from subseasonal.detrending import load_detrending_data, split_train_test
from subseasonal.metrics import weighted_rmse_step, weighted_rmse_smooth


def main(split_key="dev"):
    print(f"running persistence model for {split_key}")
    spec = SPLITS[split_key] 
    weekly_aves = read_csv_with_date(spec["data_file"]).set_index("Date")
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > spec["train_end_year"]] 

    data = load_detrending_data(split_key)
    _, test_data = split_train_test(data, spec["train_end_year"])
    
    rmse_results = []
    # Loop through forecast horizons (1 to 4 weeks ahead)
    for horizon in HORIZONS:
        for region, zscore_col in zip(REGIONS, REGION_ZSCORES):
            y_test = test_data[region]
            
            # Create the shifted predictions
            shifted_predictions = y_test.shift(horizon)
        
            z_data = weekly_aves_test[zscore_col]
            extreme_mask = (z_data[horizon:] < -1) | (z_data[horizon:] > 1)
            
            # Calculate RMSE, ignoring NaN values that result from shifting
            rmse = np.sqrt(np.mean((shifted_predictions[horizon:] - y_test[horizon:]) ** 2))
            rmse_extreme = np.sqrt(np.mean((shifted_predictions[horizon:][extreme_mask] - y_test[horizon:][extreme_mask]) ** 2))

            # Convert to numpy arrays and remove any potential NaNs jointly
            valid_mask = (~pd.isna(y_test[horizon:])) & (~pd.isna(shifted_predictions[horizon:])) & (~pd.isna(z_data[horizon:]))
            
            y_true_valid = y_test[horizon:][valid_mask]
            y_pred_valid = shifted_predictions[horizon:][valid_mask]
            z_valid = z_data[horizon:][valid_mask]

            # compute weighted scores ala Taggart 2022 (this is NOT a proper score!)
            wrmse_step = weighted_rmse_step(y_true_valid, y_pred_valid, z_valid)
            wrmse_smooth = weighted_rmse_smooth(y_true_valid, y_pred_valid, z_valid, alpha=1.0)
            
            # Store the RMSE result

            #rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse, "RMSE_extreme": rmse_extreme})
            rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse, "RMSE_extreme": rmse_extreme,
                                "Weighted_RMSE_step": wrmse_step,
                                "Weighted_RMSE_smooth": wrmse_smooth})
            
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv(Path(PERSISTENCE_DIR / split_key / f"rmse_results_persistence_{split_key}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "final"],
        default="dev",
        help="Which train/test split to run",
    )
    args = parser.parse_args()
    main(args.split)