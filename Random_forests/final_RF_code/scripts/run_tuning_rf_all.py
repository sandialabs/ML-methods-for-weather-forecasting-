import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from functools import partial

# Path setup for Slurm/HPC compatibility
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import REGIONS, HORIZONS, SPLITS
from subseasonal.features import build_lagged_feature_frame
from subseasonal.tuning import (
    load_dev_training_data, 
    run_rf_grid_search, 
    save_tuning_results, 
    extreme_scoring_function
)

def main(region, mode="standard"):
    if region not in REGIONS:
        raise ValueError(f"Invalid region '{region}'")
    
    print(f"--- Starting {mode} tuning for region: {region} ---")

    # Load data -- we always load dev for tuning
    train_data, input_feature_cols = load_dev_training_data()
    
    # Load raw data for z-scores if in extreme mode
    zscore_data = None
    if mode == "extreme":
        spec = SPLITS["dev"]
        # read_csv_with_date sets Date as index
        from subseasonal.io import read_csv_with_date
        zscore_data = read_csv_with_date(spec["data_file"]).set_index("Date")

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    tuning_results = []

    for horizon in HORIZONS:
        print(f"Tuning {horizon}-week ahead...")
        
        # Build features 
        train_subset, feature_cols, target_col = build_lagged_feature_frame(
            train_data.copy(), region, horizon, input_feature_cols
        )
        
        X_train = train_subset[feature_cols]
        y_train = train_subset[target_col]

        # Determine Scorer 
        if mode == "extreme":
            zscore_col = f"{region}_Zscore"
            z_target = zscore_data[zscore_col].shift(-horizon).loc[X_train.index]
            
            # Ensure indices align 
            if not X_train.index.equals(z_target.index):
                raise ValueError(f"Index mismatch for {region} H{horizon}")
                
            scorer = partial(extreme_scoring_function, zscore_target=z_target)
        else:
            scorer = "neg_mean_squared_error"

        # Run Search
        best_params, best_score = run_rf_grid_search(param_grid, X_train, y_train, scoring=scorer)
        
        result = {
            "Horizon": horizon, 
            "Best_Score": best_score, 
            "region": region,
            "mode": mode
        }
        result.update(best_params)
        tuning_results.append(result) 

    save_tuning_results(tuning_results, region=region, type=mode)
    print(f"✅ Tuning complete for {region} ({mode}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--mode", choices=["standard", "extreme"], default="standard")
    args = parser.parse_args()
    main(args.region, args.mode)