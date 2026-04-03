import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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


def main(mode="standard"):
    split_key = "final"
    spec = SPLITS[split_key] # this will always be final!
    train_end_year = spec["train_end_year"]
    train_data, test_data = load_train_test_data_final()

    
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
    
    # Load previously saved feature importances
    importance_df = pd.read_csv(PFI_DIR / "feature_importance.csv")  
        
        
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
            
    tuning_df = pd.concat(df_list, axis=0)

    # Dictionary to store trained models and results
    models = {}
    rmse_results = []
    RFpreds = []  # List to accumulate all results

        
    if mode == "standard":
        rmse_name = "Test_RMSE"
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
    
            allowed_features, num_features = get_best_features_with_percent_tolerance(GPFI_results, region, horizon, 
                                                                        rmse_col=rmse_name, pct_tolerance=0.01)
        
            
            X_train, y_train = build_XY_data(train_data, region, horizon, allowed_features)

            X_test, y_test = build_XY_data(test_data, region, horizon, allowed_features)
            
            # subset preds so dates match with y_test (there will be some NAs due to shifting)
            # preds from linear model on Kelvin scale
            test_y_pred = test_preds[region]
            test_y_pred_subset = test_y_pred.shift(-horizon).loc[y_test.index]
    
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
    
            # Save trained model
            models[f"{region}_{horizon}w"] = model
    
            # Predict on test set
            y_pred = model.predict(X_test)
    
            rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    
            # backtransform preds/test set targets back to Kelvin scale
            y_pred_kelvin = y_pred + test_y_pred_subset.values
            
            y_test_kelvin = weekly_aves_test[region].shift(-horizon).loc[y_test.index]
    
            # get z score column, make a mask for extremes, compute RMSE on extremes
            # must shift this by horizon b/c target is shifted by horizon
            z_data = weekly_aves_test[zscore_col].shift(-horizon).loc[y_test.index]
            extreme_mask = (z_data < -1) | (z_data > 1)
            if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
                rmse_extreme = np.sqrt(mean_squared_error(y_test.values[extreme_mask], y_pred[extreme_mask]))
            else:
                rmse_extreme = np.nan  # Assign NaN if no extremes exist
    
            # Create final dataframe for predictions
            predictions_df = pd.DataFrame({"Reference_date": y_test.index,
                                           "Observed": y_test_kelvin.values,
                                           "Predicted": y_pred_kelvin,
                                           "Horizon": horizon,
                                           "region": region})
            RFpreds.append(predictions_df)
            rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse, "RMSE_extreme": rmse_extreme,
                                 "Num_features": num_features,
                                 "Feature_names": ";".join(sorted(allowed_features))})
    
            print(f"   - RMSE: {rmse:.4f}")
            print(f"   - RMSE extreme: {rmse_extreme:.4f}")
    
    # Save RMSE results
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv(RES_DIR_FINAL / f"RF_rmse_results_{mode}_2021_2024.csv", index=False)
    
    final_preds = pd.concat(RFpreds)
    final_preds.to_csv(PRED_DIR_FINAL / f"RF_preds_{mode}_2021_2024.csv", index=False)
    
    # Print completion message
    print("All models trained and predictions saved.")  

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