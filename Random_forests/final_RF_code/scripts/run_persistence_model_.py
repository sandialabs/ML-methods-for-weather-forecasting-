from pathlib import Path
import numpy as np
import pandas as pd
import argparse
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import REGIONS, REGION_ZSCORES, PERSISTENCE_DIR
from subseasonal.io import read_csv_with_date
from subseasonal.detrending import load_detrending_data




def main(split_key="dev"):
    spec = SPLITS[split_key] 
    weekly_aves = read_csv_with_date(spec["data_file"]).set_index("Date")
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > spec["train_end_year"]] 

    data = load_detrending_data(split_key)
    _, test_data = split_train_test(data, spec["train_end_year"])

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
            
            # Store the RMSE result
            rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse, "RMSE_extreme": rmse_extreme})
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv(Path(PERSISTENCE_DIR / f"rmse_results_persistence_{dev}.csv"), index=False)

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