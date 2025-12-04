import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from functools import partial
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def make_dict():
    df = pd.read_csv("/home/mfholth/subseasonal/weekly_data/CONUS_Regions_1980_to_2020.csv")
    # Extract week number from Date column
    df['Week'] = df['Date'].astype(str).str[-2:].astype(int)
    
    # List of regions (adjust if needed)
    regions = ['SW', 'SE', 'NE', 'MW', 'W']
    
    # Initialize dictionary
    clim_stats = {}
    
    # Loop through each region and week
    for region in regions:
        for week in range(1, 53):  # Weeks 1 to 52
            values = df.loc[df['Week'] == week, region].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                clim_stats[(region, week)] = (mean, std)
    return clim_stats



def extreme_rmse_with_lookup(estimator, X, y, region, clim_stats):
    # Predict
    y_pred = estimator.predict(X)
   
    # Get week numbers from index (should be Date format YYYYWW)
    week_nums = [int(str(date)[-2:]) for date in X.index]

    # Compute z-scores using (region, week) lookups
    z_scores = []
    for val, week in zip(y, week_nums):
        mean, std = clim_stats.get((region, week), (np.nan, np.nan))
        if np.isnan(std) or std == 0:
            z_scores.append(np.nan)
        else:
            z_scores.append((val - mean) / std)

    z_scores = np.array(z_scores)
    mask = np.abs(z_scores) > 1  # or 2, depending on your definition

    if not np.any(mask):
        return -1e6  # Big penalty if no extremes found

    return -root_mean_squared_error(y[mask], y_pred[mask])

def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python rf_tuning.py <region>")
    region = sys.argv[1]

    # Allowable regions
    valid_regions = ["SW", "SE", "NE", "MW", "W"]
    if region not in valid_regions:
        raise ValueError(f"Invalid region '{region}'. Must be one of: {', '.join(valid_regions)}")

    # Load input features
    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2020.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)

    # train/test residuals from linear model
    train_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/train_residuals.csv")
    test_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/test_residuals.csv")

    # Ensure 'Date' is a string for proper merging
    for df in [regional_inputs, train_residuals, test_residuals]:
        df['Date'] = df['Date'].astype(str)

    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")

    os.makedirs("/home/mfholth/subseasonal/weekly_data/rf_tuning/", exist_ok=True)
    
    # contains dictionary of means/sd's for each week
    clim_stats = make_dict()

    # custom scoring function
    scorer = partial(extreme_rmse_with_lookup, region=region, clim_stats=clim_stats)

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    tuning_results = []

    for horizon in range(1, 5):
        print(f"Tuning hyperparameters for {region} - {horizon}-week ahead", flush=True)

        train_data[f"{region}_target"] = train_data[region].shift(-horizon)
        test_data[f"{region}_target"] = test_data[region].shift(-horizon)

        train_data[f"{region}_current_week"] = train_data[region]
        test_data[f"{region}_current_week"] = test_data[region]

        for lag in range(1, 6):
            train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)
            test_data[f"{region}_lag_{lag}"] = test_data[region].shift(lag)

        train_subset = train_data.dropna(subset=[f"{region}_target"])
        test_subset = test_data.dropna(subset=[f"{region}_target"])

        feature_cols = [f"{region}_current_week"] + [col for col in regional_inputs.columns if col != "Date"]
        X_train = train_subset[feature_cols]
        y_train = train_subset[f"{region}_target"]

        #tscv = TimeSeriesSplit(n_splits=5)
        rf = RandomForestRegressor(random_state=42, n_jobs=1)
        grid_search = GridSearchCV(rf, param_grid, cv=4, scoring=scorer, n_jobs=10, verbose=1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        result = {"Horizon": horizon, "Best RMSE": best_score, "region": region}
        result.update(best_params) 
    
        tuning_results.append(result)

        print(f"   - Best RMSE: {best_score:.4f}", flush=True)
        print(f"   - Best Parameters: {best_params}", flush=True)

    tuning_results_df = pd.DataFrame(tuning_results)
    tuning_results_df.to_csv(f"/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_{region}_EXTREMES_cv.csv", index=False)
    print("\n✅ Hyperparameter tuning complete. Results saved.", flush=True)

if __name__ == "__main__":
    main()

    