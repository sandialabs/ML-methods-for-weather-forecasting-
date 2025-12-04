import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from copy import deepcopy
#import networkx as nx
from collections import defaultdict


def train_and_evaluate_model(train_data, test_data, tuning_df, selected_features, region, horizon, zscore_col, weekly_aves_test):

    def fill_or_default(val, default):
        return default if pd.isna(val) else val
    
    def parse_maxdepth(val):
        if pd.isna(val):
            return None
        return int(val)
    
    row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
    
    train_data[f"{region}_target"] = train_data[region].shift(-horizon)
    test_data[f"{region}_target"] = test_data[region].shift(-horizon)
    
    # Create 'current week' predictor (target variable at lag 0)
    train_data[f"{region}_current_week"] = train_data[region]  # No shift
    test_data[f"{region}_current_week"] = test_data[region]    # No shift
    
    # Create up to 5 lagged predictors
    for lag in range(1, 6):  # Lags 1 to 5
        train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)
        test_data[f"{region}_lag_{lag}"] = test_data[region].shift(lag)
    
    # Drop rows with NaN targets (last few weeks in training set)
    train_subset = train_data.dropna(subset=[f"{region}_target"])
    test_subset = test_data.dropna(subset=[f"{region}_target"])
    
    # Define features (include 'current week' as a predictor)
    feature_cols = selected_features
    
    X_train = train_subset[feature_cols]
    y_train = train_subset[f"{region}_target"]
    X_test = test_subset[feature_cols]
    y_test = test_subset[f"{region}_target"]

    weekly_aves_test_subset = weekly_aves_test.loc[y_test.index,:]
    
            # Train Random Forest Model
    #model = RandomForestRegressor(n_estimators=300, max_depth=20,
    #                              random_state=42,min_samples_leaf=2,
    #                              min_samples_split=2,
    #                              n_jobs=-1)
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
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    
    z_data = weekly_aves_test_subset[zscore_col]
    extreme_mask = (z_data < -1) | (z_data > 1)
    
    if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
        rmse_extreme = np.sqrt(mean_squared_error(y_test[extreme_mask], y_pred[extreme_mask]))
    else:
        rmse_extreme = np.nan
    
    return rmse, rmse_extreme


def main():
    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2020.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)

    weekly_aves = pd.read_csv("/home/mfholth/subseasonal/weekly_data/CONUS_Regions_1980_to_2020.csv")
    weekly_aves['Date'] = weekly_aves['Date'].astype(str)
    weekly_aves.set_index('Date', inplace=True)
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > 2016]

    
    # train/test residuals from linear model 
    train_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/train_residuals.csv")
    test_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/test_residuals.csv")
    
    # Ensure 'Date' is a string for proper merging
    regional_inputs['Date'] = regional_inputs['Date'].astype(str)
    train_residuals['Date'] = train_residuals['Date'].astype(str)
    test_residuals['Date'] = test_residuals['Date'].astype(str)
    
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    
    # Preprocess to filter out negative importances
    #importance_df = importance_df[importance_df['Importance'] >= 0]
    
    file_list = ['/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_MW_cv.csv', 
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_NE_cv.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SE_cv.csv', 
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SW_cv.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_W_cv.csv']

    
    # TS tuned parameters
    #file_list = ['/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_MW_EXTREMES.csv', 
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_NE_EXTREMES.csv',
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SE_EXTREMES.csv', 
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SW_EXTREMES.csv',
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_W_EXTREMES.csv']
    
    
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
        
    tuning_df = pd.concat(df_list, axis=0)


    # Load grouped PFI results
    gpfi_df = pd.read_csv("/home/mfholth/subseasonal/weekly_data/Grouped_PFI_datasets/All_GPFI_Merged_RF_60.csv")
    
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
        used_features = set()
    
        for i in range(1, len(group_means) + 1):
            top_groups = group_means.iloc[:i]["Group"].values
            selected_features = group_data[group_data["Group"].isin(top_groups)]["Feature"].unique()
           
            rmse, rmse_extreme = train_and_evaluate_model(train_data, test_data, tuning_df, selected_features, region, horizon, zscore_col, weekly_aves_test)
    
            # Save results
            results.append({
                "Region": region,
                "Horizon": horizon,
                "Num_Groups": i,
                "Num_Features": len(selected_features),
                "Test_RMSE": rmse,
                "Test_RMSE_extreme": rmse_extreme
            })
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("/home/mfholth/subseasonal/weekly_data/rf_feature_importance/iterative_group_rmse_results_RF_60.csv", index=False)

    print("All done.")

if __name__ == "__main__":
    main()
