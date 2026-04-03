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

from subseasonal.config import REGIONS, HORIZONS, TUNING_DIR, RAW_DIR, PFI_DIR, REDUNDANT_COLS, SPLITS
from subseasonal.GPFI import load_train_test_data_dev, train_and_evaluate_model
from subseasonal.io import read_csv_with_date

def main(mode="standard"):

    spec = SPLITS["dev"] # this will always be dev!
    train_end_year = spec["train_end_year"]
    train_data, test_data = load_train_test_data_dev()

    weekly_aves = read_csv_with_date(spec["data_file"]).set_index("Date")
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > train_end_year] # is 2016

    # TS tuned parameters
    file_list = [Path(TUNING_DIR / mode / f"rf_tuning_results_MW_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_NE_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_SE_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_SW_tscv_{mode}.csv"),
                    Path(TUNING_DIR / mode / f"rf_tuning_results_W_tscv_{mode}.csv")]
  
    
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
        
    tuning_df = pd.concat(df_list, axis=0)


    # Load grouped PFI results
    gpfi_df = pd.read_csv(PFI_DIR / f"All_GPFI_Merged_RF_TS_{mode}.csv")
    
    # Filter to positive importance only
    gpfi_df = gpfi_df[gpfi_df["Mean_Importance"] > 0]
    
    # Initialize output dictionary
    results = []
    
    # Loop over all region/horizon combinations
    for (region, horizon), group_data in gpfi_df.groupby(["Region", "Horizon"]):
    
        print(f"Processing Region: {region}, Horizon: {horizon}", flush=True)
       
        # Sort groups by decreasing importance (mean across features in group)
        group_means = (
            group_data.groupby("Group")
            .agg({"Mean_Importance": "mean"})
            .sort_values("Mean_Importance", ascending=False)
            .reset_index()
        )

        zscore_col = f"{region}_Zscore"
    
        for i in range(1, len(group_means) + 1):
            top_groups = group_means.iloc[:i]["Group"].values
            selected_features = group_data[group_data["Group"].isin(top_groups)]["Feature"].unique()
            selected_features = list(selected_features)
           
            rmse, rmse_extreme = train_and_evaluate_model(train_data, test_data, tuning_df, selected_features, region, horizon, zscore_col, weekly_aves_test)
    
            # Save results
            results.append({
                "Region": region,
                "Horizon": horizon,
                "Num_Groups": i,
                "Num_Features": len(selected_features),
                "Test_RMSE": rmse,
                "Test_RMSE_extreme": rmse_extreme,
                "Feature_names": ";".join(sorted(selected_features))
            })
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(PFI_DIR / f"iterative_group_rmse_results_RF_TS_{mode}.csv", index=False)

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme"],
        default="standard",
        help="run standard (standard) or only on extremes (extreme)",
    )
    args = parser.parse_args()
    main(args.mode)