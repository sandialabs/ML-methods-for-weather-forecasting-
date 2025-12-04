import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn_quantile import RandomForestQuantileRegressor

def get_best_features_for_region_horizon(results_df, region, horizon, rmse_col="Test_RMSE", tolerance=0.01):
    """
    Return the list of features corresponding to the best model
    for a given Region and Horizon from the results_df.
   
    Parameters:
    - results_df: pandas DataFrame containing columns ["Region", "Horizon", "Test_RMSE", "Feature_Names"]
    - region: str, e.g., "MW"
    - horizon: int, e.g., 1
    - rmse_col: which column to consider for best RMSE (can be "Test_RMSE_extreme" too)
    - tolerance: float, RMSE tolerance within which model is still considered 'best'
   
    Returns:
    - List of feature names (strings)
    """
   
    # Filter to the right region and horizon
    sub_df = results_df[(results_df["Region"] == region) & (results_df["Horizon"] == horizon)]
   
    # Find minimum RMSE in that group
    min_rmse = sub_df[rmse_col].min()
   
    # Find models within tolerance
    best_rows = sub_df[sub_df[rmse_col] <= min_rmse + tolerance]
   
    # Take the first such row (in case of tie)
    best_row = best_rows.iloc[0]
   
    # Parse features (assuming they were saved as semi-colon separated string)
    features_str = best_row["Feature_names"]
    num_features = best_row["Num_Features"]
    selected_features = [f.strip() for f in features_str.split(";") if f.strip()]
   
    return selected_features, num_features

def get_best_features_with_percent_tolerance(results_df, region, horizon, rmse_col="Test_RMSE", pct_tolerance=0.01):
    """
    Return the list of features corresponding to the best model
    within a percentage tolerance of the minimum RMSE.
   
    Parameters:
    - results_df: pandas DataFrame with ["Region", "Horizon", rmse_col, "Feature_Names"]
    - region: str, e.g., "MW"
    - horizon: int, e.g., 1
    - rmse_col: str, the RMSE column to use (e.g. "Test_RMSE" or "Test_RMSE_extreme")
    - pct_tolerance: float, e.g., 0.01 for 1%
   
    Returns:
    - List of feature names (strings)
    """

    sub_df = results_df[(results_df["Region"] == region) & (results_df["Horizon"] == horizon)]
    min_rmse = sub_df[rmse_col].min()
   
    # Compute upper bound as (1 + pct_tolerance) × min_rmse
    tolerance_threshold = min_rmse * (1 + pct_tolerance)
   
    best_rows = sub_df[sub_df[rmse_col] <= tolerance_threshold]
    best_row = best_rows.iloc[0]
   
    features_str = best_row["Feature_names"]
    num_features = best_row["Num_Features"]
    selected_features = [f.strip() for f in features_str.split(";") if f.strip()]
   
    return selected_features, num_features

# functions for filling in tuning params for RF
def fill_or_default(val, default):
    return default if pd.isna(val) else val

def parse_maxdepth(val):
    if pd.isna(val):
        return None
    return int(val)

# NOTE we train on ALL data 1980-2020 here and predict onto 2021-2024
def main():
    results_folder = "results_TEST"
    # use this for modeling -- read in inputs/targets/preds from linear model for backtransforming
    # Load input features
    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2024.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)
    #regional_inputs = regional_inputs.drop(columns = redundant_cols)

    data=pd.read_csv("/home/mfholth/subseasonal/weekly_data/CONUS_Regions_1980_to_2024.csv")
    data['Date'] = data['Date'].astype(str)
    data.set_index('Date', inplace=True)
    y_test_data = data.loc[data.index.str[:4].astype(int) > 2020]
    
    # train/test residuals from linear model 
    train_residuals = pd.read_csv(os.path.join(results_folder, "train_residuals.csv"))
    test_residuals = pd.read_csv(os.path.join(results_folder, "test_residuals.csv"))
    
    # preds from linear model on train set --  for backtransforming
    train_preds = pd.read_csv(os.path.join(results_folder, "train_preds.csv"))
    train_preds['Date'] = train_preds['Date'].astype(str)
    train_preds.set_index('Date', inplace=True)
    # preds from linear model on test set -- for backtransforming
    test_preds = pd.read_csv(os.path.join(results_folder, "test_preds.csv"))
    test_preds['Date'] = test_preds['Date'].astype(str)
    test_preds.set_index('Date', inplace=True)
    # has Z scores
    weekly_aves = pd.read_csv("CONUS_Regions_1980_to_2024.csv")
    weekly_aves['Date'] = weekly_aves['Date'].astype(str)
    weekly_aves.set_index('Date', inplace=True)
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > 2020] # was 2016, this is now 2020
    
    # Ensure 'Date' is a string for proper merging
    regional_inputs['Date'] = regional_inputs['Date'].astype(str)
    train_residuals['Date'] = train_residuals['Date'].astype(str)
    test_residuals['Date'] = test_residuals['Date'].astype(str)
    
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    
    # Define regions
    regions = [col for col in train_residuals.columns if col != "Date"]
    regions = ['W', 'SW', 'MW', 'SE', 'NE']  # Update if there are more
    region_zscores = [f"{region}_Zscore" for region in regions]
    
    regional_inputs.set_index("Date")

    # grouped PFI results
    GPFI_results = pd.read_csv('/home/mfholth/subseasonal/weekly_data/rf_feature_importance/iterative_group_rmse_results_RF_TS.csv')

    # TS tuned parameters
    file_list = ['/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_MW.csv', 
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_NE.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SE.csv', 
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SW.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_W.csv']

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
        
    tuning_df = pd.concat(df_list, axis=0)
    
    # lists to store preds/results
    RFpreds = []  # List to accumulate all results
    rmse_results = []
    
    # Ensure output directories exist
    os.makedirs("rf_predictions", exist_ok=True)

    
    # Train models for 1-4 week-ahead forecasting
    for horizon in range(1, 5):  # 1 to 4 weeks ahead
        print(f"Training models for {horizon}-week ahead forecasting...")
       
        for region, zscore_col in zip(regions, region_zscores):
            print(f" - Processing region: {region}")
            row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
    
            train_data[f"{region}_target"] = train_data[region].shift(-horizon)
            test_data[f"{region}_target"] = test_data[region].shift(-horizon)

            # only use features from the 'best models'
            allowed_features, num_features = get_best_features_with_percent_tolerance(GPFI_results, region, horizon, 
                                                                    rmse_col="Test_RMSE", pct_tolerance=0.01)
 
    
            # Create 'current week' predictor (target variable at lag 0)
            train_data[f"{region}_current_week"] = train_data[region]  # No shift
            test_data[f"{region}_current_week"] = test_data[region]    # No shift
    
            # Create up to 5 lagged predictors
            for lag in range(1, 6):  # Lags 1 to 5
                train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)
                test_data[f"{region}_lag_{lag}"] = test_data[region].shift(lag)
    
            # Drop rows with NaN targets 
            train_subset = train_data.dropna(subset=[f"{region}_target"])
            test_subset = test_data.dropna(subset=[f"{region}_target"])
    
            test_y_pred = test_preds[region]
    
            # Define features (include 'current week' as a predictor)
            lagged_features = [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, 6)]
            feature_cols = lagged_features + [f for f in allowed_features if f not in lagged_features]

            
            X_train = train_subset[feature_cols]
            y_train = train_subset[f"{region}_target"]
            X_test = test_subset[feature_cols]
            test_set_indices = X_test.index
            
            y_test = y_test_data[region]
            y_test = y_test[test_set_indices]
            
    
            # subset preds so dates match with X_test (test dates are always 2021-2024)
            # but since we use lagged predictors we lose a few obs at the end of the time series
            test_y_pred_subset = test_y_pred.loc[test_set_indices]
    
            # Subset Z score dataset so dates match with y_test
            weekly_aves_test_subset = weekly_aves_test.loc[test_set_indices,:]
    
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
    
            # Predict on test set
            y_pred = model.predict(X_test)
    
            # backtransform preds/test set targets back to Kelvin scale, compute RMSE
            y_pred_kelvin = y_pred + test_y_pred_subset.values
            y_test_kelvin = y_test.values 
            rmse = np.sqrt(mean_squared_error(y_test_kelvin, y_pred_kelvin))

            # Create final dataframe for predictions
            predictions_df = pd.DataFrame({"Date": y_test.index, "Observed": y_test_kelvin, "Predicted": y_pred_kelvin,
                                          "Horizon": horizon, "region": region})

            z_data = weekly_aves_test_subset[zscore_col]
            predictions_df["is_extreme"] = ((z_data < -1) | (z_data > 1)).astype(int)

            RFpreds.append(predictions_df)
    
            # get z score column, make a mask for extremes, compute RMSE on extremes
            z_data = weekly_aves_test_subset[zscore_col]
            extreme_mask = (z_data < -1) | (z_data > 1)
            if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
                rmse_extreme = np.sqrt(mean_squared_error(y_test_kelvin[extreme_mask], y_pred_kelvin[extreme_mask]))
            else:
                rmse_extreme = np.nan  # Assign NaN if no extremes exist
            rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse, "RMSE_extreme": rmse_extreme,
                                 "Num_features": num_features,
                                 "Feature_names": ";".join(sorted(allowed_features))})
            
    
            print(f"   - RMSE: {rmse:.4f}")
            print(f"   - RMSE extreme: {rmse_extreme:.4f}")
    
    # Save RMSE results
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv("/home/mfholth/subseasonal/weekly_data/results_TEST/results2021_2024/RF_rmse_results_ChooseEXTREME.csv", index=False)
    #rmse_df.to_csv("/home/mfholth/subseasonal/weekly_data/results_TEST/results2021_2024/RF_rmse_results.csv", index=False)

    final_preds = pd.concat(RFpreds)
    final_preds.to_csv("/home/mfholth/subseasonal/weekly_data/results_TEST/results2021_2024/RF_preds_ChooseEXTREME.csv", index=False)
    #final_preds.to_csv("/home/mfholth/subseasonal/weekly_data/results_TEST/results2021_2024/RF_preds.csv", index=False)
    
    # Print completion message
    print("All models trained and predictions saved.")

if __name__ == "__main__":
    main()


    