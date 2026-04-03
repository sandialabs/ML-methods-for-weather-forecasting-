# src/subseasonal/features.py
import pandas as pd
from .config import MAX_LAG, SPLITS, DETREND_OUTPUTS, REDUNDANT_COLS
from .io import read_csv_with_date

def build_lagged_feature_frame(
    data: pd.DataFrame,
    region: str,
    horizon: int, 
    input_feature_cols: list[str]
):
    target_col = f"{region}_target"
    # Target at forecast week t+h, aligned to reference week t
    data[target_col] = data[region].shift(-horizon)

    # Current week + lagged residual predictors
    data[f"{region}_current_week"] = data[region]
    for lag in range(1, MAX_LAG):
        data[f"{region}_lag_{lag}"] = data[region].shift(lag)

    lagged_features = [f"{region}_current_week"] + [f"{region}_lag_{lag}" for lag in range(1, MAX_LAG)]

    # Remove any accidental duplicate of current_week if region column is already in inputs
    feature_cols = lagged_features + [col for col in input_feature_cols if col != "Date" and col not in lagged_features]

    # Drop rows with missing target or missing predictors
    drop_cols = [target_col] + feature_cols
    train_subset = data.dropna(subset=drop_cols).copy()
    return train_subset, feature_cols, target_col

def load_train_test_data_final() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPLITS["final"]
    resid_spec = DETREND_OUTPUTS["final"]
    regional_inputs = read_csv_with_date(spec["inputs_file"])
    
    # drop columns deemed to be redundant from correlation analysis
    regional_inputs = regional_inputs.drop(columns = REDUNDANT_COLS)
    
    train_residuals = read_csv_with_date(resid_spec["train_residuals_file"])
    test_residuals = read_csv_with_date(resid_spec["test_residuals_file"])
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    return train_data, test_data

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



