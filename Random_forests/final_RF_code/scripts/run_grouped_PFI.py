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
from subseasonal.GPFI import build_XY_data, grouped_pfi, get_clusters, load_train_test_data_dev, save_final_GPFI_dataset
from subseasonal.tuning import fill_or_default, parse_maxdepth


def main(mode="standard"):
    print(f"Running grouped PFI for mode {mode}")
    # use this for modeling -- read in inputs/targets/preds from linear model for backtransforming
    # Load input features
    train_data, test_data = load_train_test_data_dev()
    
    # Load previously saved feature importances
    # THIS IS ONLY FOR GETTING THE CORRECT FEATURES FOR EACH MODEL
    importance_df = pd.read_csv(Path(PFI_DIR / "feature_importance.csv"))  
    
    # tuning files
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
    
    # Train models for 1-4 week-ahead forecasting
    for horizon in HORIZONS:  # 1 to 4 weeks ahead
        print(f"Running group PFI for {horizon}-week ahead forecasting...")
       
        for region in REGIONS:
            print(f" - Processing region: {region}")
            row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
            candidate_features = importance_df[(importance_df['Region'] == region) & (importance_df['Horizon'] == horizon)]['Feature'].tolist()

            X_train, y_train = build_XY_data(train_data, region, horizon, candidate_features)

            X_test, y_test = build_XY_data(test_data, region, horizon, candidate_features)
    
    
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

            cluster_df = get_clusters(horizon, region)
            cluster_map = dict(zip(cluster_df["Feature"], cluster_df["Cluster"]))
            
            missing = [col for col in X_test.columns if col not in cluster_map]
            if missing:
                raise ValueError(f"Missing cluster assignments for: {missing[:10]}")
            
            group_dict = {}
            for col in X_test.columns:
                group_id = cluster_map[col]
                group_dict.setdefault(group_id, []).append(col)

            group_importance_df = grouped_pfi(model, X_test, y_test, group_dict, metric=mean_squared_error, n_repeats=20)
            
            # Flatten the dictionary into a list of (group_id, variable) pairs
            group_data = [(group_id, var) for group_id, vars in group_dict.items() for var in vars]
            
            # Convert to DataFrame
            group_df = pd.DataFrame(group_data, columns=['Group', 'Feature'])
            
            # sort for easier inspection
            group_df = group_df.sort_values(by='Group').reset_index(drop=True)
            
            group_importance_df.to_csv(Path(PFI_DIR / f'GPFI_{region}_{horizon}_{mode}.csv'), index = False)
            group_df.to_csv(Path(PFI_DIR / f'Groups_{region}_{horizon}_{mode}.csv'), index = False)

            train_data, test_data = load_train_test_data_dev()
    
    # Print completion message
    print("All PFI done, saving final file.")
    save_final_GPFI_dataset(mode=mode)
    print(f"Results saved to {PFI_DIR}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme"],
        default="standard",
        help="Type of inference--extreme or all (standard)",
    )
    args = parser.parse_args()
    main(args.mode)
