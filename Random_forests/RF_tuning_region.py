import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 20, 50, None],
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
        grid_search = GridSearchCV(rf, param_grid, cv=4, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        result = {"Horizon": horizon, "Best RMSE": best_score, "region": region}
        result.update(best_params) 
    
        tuning_results.append(result)

        print(f"   - Best RMSE: {best_score:.4f}", flush=True)
        print(f"   - Best Parameters: {best_params}", flush=True)

    tuning_results_df = pd.DataFrame(tuning_results)
    tuning_results_df.to_csv(f"/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_{region}_cv.csv", index=False)
    print("\n✅ Hyperparameter tuning complete. Results saved.", flush=True)

if __name__ == "__main__":
    main()