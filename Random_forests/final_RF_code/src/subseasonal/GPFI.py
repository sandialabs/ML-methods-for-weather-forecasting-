# /src/subseasonal/GPFI.py
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from pathlib import Path
import pandas as pd
import numpy as np

from .io import read_csv_with_date, ensure_dir
from .config import SPLITS, PROJECT_ROOT, DETREND_OUTPUTS, TUNING_DIR, RAW_DIR, REDUNDANT_COLS, MAX_LAG, REGIONS, HORIZONS
from .tuning import fill_or_default, parse_maxdepth


def load_train_test_data_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPLITS["dev"]
    resid_spec = DETREND_OUTPUTS["dev"]
    regional_inputs = read_csv_with_date(spec["inputs_file"])
    
    # drop columns deemed to be redundant from correlation analysis
    regional_inputs = regional_inputs.drop(columns = REDUNDANT_COLS)
    
    train_residuals = read_csv_with_date(resid_spec["train_residuals_file"])
    test_residuals = read_csv_with_date(resid_spec["test_residuals_file"])
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    return train_data, test_data

def build_XY_data(
    data: pd.DataFrame,
    region: str,
    horizon: int,
    candidate_features: list[str]
    
):
    data[f"{region}_target"] = data[region].shift(-horizon)
    
    # this is only for getting features
    candidate_features = [feat for feat in candidate_features if feat not in REDUNDANT_COLS]
    
    # Create 'current week' predictor (target variable at lag 0)
    data[f"{region}_current_week"] = data[region] # No shift

    for lag in range(1, MAX_LAG):  # Lags 1 to 5
        data[f"{region}_lag_{lag}"] = data[region].shift(lag)

    # Define features (include 'current week' as a predictor)
    lagged_features = [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, MAX_LAG)]
    feature_cols = lagged_features + [f for f in candidate_features if f not in lagged_features]

    # Drop rows with NaN targets (last few weeks in training set)
    data_subset = data.dropna(subset=[f"{region}_target"] + feature_cols)


    X = data_subset[feature_cols]
    y = data_subset[f"{region}_target"]

    return X, y

def grouped_pfi(model, X_test, y_test, group_dict, metric=mean_squared_error, n_repeats=5, random_state=42):
    """
    Compute grouped permutation feature importance.

    Parameters:
        model : trained sklearn regressor
        X_test : pd.DataFrame
        y_test : pd.Series or np.array
        group_dict : dict, {group_id: [list of feature names]}
        metric : function, error metric (e.g., mean_squared_error)
        n_repeats : int, number of permutations per group
        random_state : int, for reproducibility

    Returns:
        pd.DataFrame with group_id, mean_importance, and std
    """
    rng = np.random.RandomState(random_state)
    base_pred = model.predict(X_test)
    base_error = metric(y_test, base_pred)

    results = []

    for group_id, features in group_dict.items():
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            for feat in features:
                X_perm[feat] = rng.permutation(X_perm[feat].values)
            y_perm = model.predict(X_perm)
            perm_error = metric(y_test, y_perm)
            scores.append(perm_error - base_error)

        results.append({
            "Group": group_id,
            "Mean_Importance": np.mean(scores),
            "Std_Importance": np.std(scores),
            "Group_Size": len(features)
        })

    return pd.DataFrame(results).sort_values("Mean_Importance", ascending=False)

def get_clusters(horizon, region):
    '''
    Function to determine clusters for PFI using hierarchical clustering

    Returns
    pandas Dataframe with inputs and cluster groups
    '''
    corr_matrix = pd.read_csv(Path(RAW_DIR / "correlation_matrices" / f"corr_{region}_h{horizon}.csv"))
    corr_matrix = corr_matrix.drop(columns='Unnamed: 0')

    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    # Convert to condensed format for linkage (required format)
    condensed_dist = squareform(distance_matrix.values)
    
    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Choose a threshold to cut the tree and form clusters
    threshold = 0.7  # seems to work reasonably well 
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Create DataFrame showing variable-to-cluster mapping
    cluster_df = pd.DataFrame({
        'Feature': corr_matrix.columns,
        'Cluster': cluster_labels
    }).sort_values(by='Cluster')

    return cluster_df


def train_and_evaluate_model(train_data,
                             test_data,
                             tuning_df,
                             selected_features,
                             region,
                             horizon,
                             zscore_col, 
                             weekly_aves_test):
    

    row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
    
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    train_data[f"{region}_target"] = train_data[region].shift(-horizon)
    test_data[f"{region}_target"] = test_data[region].shift(-horizon)
    
    # Create 'current week' predictor (target variable at lag 0)
    train_data[f"{region}_current_week"] = train_data[region]  # No shift
    test_data[f"{region}_current_week"] = test_data[region]    # No shift
    
    # Create up to 5 lagged predictors
    for lag in range(1, MAX_LAG):  # Lags 1 to 5
        train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)
        test_data[f"{region}_lag_{lag}"] = test_data[region].shift(lag)
    
    # Drop rows with NaN targets (last few weeks in training set)
    train_subset = train_data.dropna(subset=[f"{region}_target"] + selected_features)
    test_subset = test_data.dropna(subset=[f"{region}_target"] + selected_features)
    
    X_train = train_subset[selected_features]
    y_train = train_subset[f"{region}_target"]
    X_test = test_subset[selected_features]
    y_test = test_subset[f"{region}_target"]

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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    z_data = weekly_aves_test[zscore_col].shift(-horizon).loc[y_test.index]
    extreme_mask = (z_data < -1) | (z_data > 1)
    
    if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
        rmse_extreme = np.sqrt(mean_squared_error(y_test[extreme_mask], y_pred[extreme_mask]))
    else:
        rmse_extreme = np.nan
    
    return rmse, rmse_extreme



def save_final_GPFI_dataset(mode):
    # List to hold the merged DataFrames
    merged_list = []
    
    for region in REGIONS:
        for horizon in HORIZONS:
            # File paths
            gpfi_file = Path(PFI_DIR / f'GPFI_{region}_{horizon}_{mode}.csv')  
            groups_file = Path(PFI_DIR / f'Groups_{region}_{horizon}_{mode}.csv') 
           
            # Read the files
            try:
                gpfi_df = pd.read_csv(gpfi_file)
                groups_df = pd.read_csv(groups_file)
    
                # Merge on 'Group'
                merged_df = pd.merge(gpfi_df, groups_df, on='Group', how='left')
    
                # Add region and horizon columns for filtering
                merged_df['Region'] = region
                merged_df['Horizon'] = horizon
    
                # Append to list
                merged_list.append(merged_df)
    
            except FileNotFoundError as e:
                print(f"Skipping {region}-{horizon} due to missing file: {e}")
    
    # Combine all merged DataFrames
    final_df = pd.concat(merged_list, ignore_index=True)
    
    # Optional: Save to a new file
    final_df.to_csv(Path(PFI_DIR / f"All_GPFI_Merged_RF_TS_{mode}.csv"), index=False)
    
