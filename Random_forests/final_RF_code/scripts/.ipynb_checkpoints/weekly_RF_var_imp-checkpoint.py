import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# this script is only for calculating permutation feature importance

def main():
    # Load input features
    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2020.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)
    #regional_inputs = regional_inputs.drop(columns = redundant_cols)
    
    # train/test residuals from linear model 
    train_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/train_residuals.csv")
    test_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/test_residuals.csv")
    
    # preds from linear model on train set --  for backtransforming
    train_preds = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/train_preds.csv")
    train_preds['Date'] = train_preds['Date'].astype(str)
    train_preds.set_index('Date', inplace=True)
    # preds from linear model on test set -- for backtransforming
    test_preds = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/test_preds.csv")
    test_preds['Date'] = test_preds['Date'].astype(str)
    test_preds.set_index('Date', inplace=True)
    # has Z scores
    weekly_aves = pd.read_csv("/home/mfholth/subseasonal/weekly_data/CONUS_Regions_1980_to_2020.csv")
    weekly_aves['Date'] = weekly_aves['Date'].astype(str)
    weekly_aves.set_index('Date', inplace=True)
    weekly_aves_test = weekly_aves.loc[weekly_aves.index.str[:4].astype(int) > 2016]
    
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
    
    # Dictionary to store trained models and results
    models = {}
    rmse_results = []
    feature_importances = []
    
    # Ensure output directories exist
    #os.makedirs("/home/mfholth/subseasonal/weekly_data/rf_models", exist_ok=True)
    #os.makedirs("/home/mfholth/subseasonal/weekly_data/rf_predictions", exist_ok=True)
    os.makedirs("/home/mfholth/subseasonal/weekly_data/rf_feature_importance", exist_ok=True)
    
    # Train models for 1-4 week-ahead forecasting
    for horizon in range(1, 5):  # 1 to 4 weeks ahead
        print(f"Training models for {horizon}-week ahead forecasting...")
       
        for region in regions:
            print(f" - Processing region: {region}")
    
            # Create lagged targets for forecasting
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
            feature_cols = [f"{region}_current_week"] + [col for col in regional_inputs.columns if col != "Date"]
            X_train = train_subset[feature_cols]
            y_train = train_subset[f"{region}_target"]
            X_test = test_subset[feature_cols]
            y_test = test_subset[f"{region}_target"]
    
            # Train Random Forest Model
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
    
            # Compute permutation importance
            print(f"   - Calculating feature importance for {region} ({horizon} weeks ahead)...")
            perm_importance = permutation_importance(model, X_test, y_test, scoring="neg_mean_squared_error", n_repeats=30, random_state=42, n_jobs=-1)
    
            # Store feature importances
            for feature, importance in zip(feature_cols, perm_importance.importances_mean):
                feature_importances.append({
                    "Region": region,
                    "Horizon": horizon,
                    "Feature": feature,
                    "Importance": importance
                })
    
            # Save trained model
            # models[f"{region}_{horizon}w"] = model
    
            # Predict on test set
            # y_pred = model.predict(X_test)
    
            # Compute RMSE
            #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #rmse_results.append({"Region": region, "Horizon": horizon, "RMSE": rmse})
    
            # Save predictions
            #predictions_df = pd.DataFrame({"Date": test_subset["Date"], "Observed": y_test, "Predicted": y_pred})
            #predictions_df.to_csv(f"/home/mfholth/subseasonal/weekly_data/rf_predictions/{region}_{horizon}w_predictions.csv", index=False)
    
            #print(f"   - RMSE: {rmse:.4f}")
    
    # Save RMSE results
    #rmse_df = pd.DataFrame(rmse_results)
    #rmse_df.to_csv("/home/mfholth/subseasonal/weekly_data/rf_predictions/rmse_results.csv", index=False)
    
    # Save feature importances
    feature_importance_df = pd.DataFrame(feature_importances)
    feature_importance_df.to_csv("/home/mfholth/subseasonal/weekly_data/rf_feature_importance/feature_importance.csv", index=False)

    # Print completion message
    print("done.")

if __name__ == "__main__":
    main()