import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import os
from copy import deepcopy


def get_clusters(horizon, region):
    '''
    Function to determine clusters for PFI using hierarchical clustering

    Returns
    pandas Dataframe with inputs and cluster groups
    '''
    corr_matrix = pd.read_csv(f'/home/mfholth/subseasonal/weekly_data/correlation_matrices/corr_{region}_h{horizon}.csv')
    corr_matrix = corr_matrix.drop(columns='Unnamed: 0')

    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    # Convert to condensed format for linkage (required format)
    condensed_dist = squareform(distance_matrix.values)
    
    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Choose a threshold to cut the tree and form clusters
    threshold = 0.6  # seems to work reasonably well 
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Create DataFrame showing variable-to-cluster mapping
    cluster_df = pd.DataFrame({
        'Variable': corr_matrix.columns,
        'Cluster': cluster_labels
    }).sort_values(by='Cluster')

    return cluster_df
    

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

# functions for filling in tuning params for RF
def fill_or_default(val, default):
    return default if pd.isna(val) else val

def parse_maxdepth(val):
    if pd.isna(val):
        return None
    return int(val)


def main():
    os.makedirs("/home/mfholth/subseasonal/weekly_data/Grouped_PFI", exist_ok=True)

    redundant_cols = ['slp_weekave_atlantic_ocean_mean', 'h_850_weekave_conus_mean', 'h_850_weekave_atlantic_ocean_pc4',
                  'h_850_weekave_mexico_gulf_mean', 'slp_weekave_mexico_gulf_mean', 'slp_weekave_conus_pc1', 'ts_weekave_conus_pc1', 
                  'slp_weekave_atlantic_trop_mean', 'ts_weekave_southern_canada_pc1', 't_850_weekave_southern_canada_mean', 
                  'h_850_weekave_atlantic_ocean_pc5', 'slp_weekave_atlantic_ocean_pc6', 'h_500_weekave_pacific_trop_mean', 
                  'h_850_weekave_atlantic_trop_mean', 't_500_weekave_pacific_trop_mean', 't_500_weekave_atlantic_trop_pc1',
                  'h_850_weekave_pacific_ocean_pc1', 't_850_weekave_conus_pc1', 'slp_weekave_atlantic_ocean_pc1', 
                  'slp_weekave_pacific_trop_mean', 't_850_weekave_southern_canada_pc1', 'h_850_weekave_arctic_pc1',
                  'h_500_weekave_atlantic_trop_mean', 'h_850_weekave_atlantic_ocean_pc2', 'slp_weekave_atlantic_ocean_pc3',
                  'h_850_weekave_atlantic_ocean_pc7', 't_500_weekave_mexico_gulf_mean', 't_500_weekave_southern_canada_pc1', 
                  'slp_weekave_pacific_trop_pc2', 'h_850_weekave_arctic_mean', 'ts_weekave_atlantic_trop_mean',
                  'h_850_weekave_pacific_trop_mean']
    # use this for modeling -- read in inputs/targets/preds from linear model for backtransforming
    # Load input features
    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2020.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)
    regional_inputs = regional_inputs.drop(columns = redundant_cols)
    
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
    
    # Define regions
    regions = ['W', 'SW', 'MW', 'SE', 'NE'] 
    #regions = ['SW'] 
    
    regional_inputs.set_index("Date")
    
    # Load previously saved feature importances
    # THIS IS ONLY FOR GETTING THE CORRECT FEATURES FOR EACH MODEL
    importance_df = pd.read_csv("/home/mfholth/subseasonal/weekly_data/rf_feature_importance/feature_importance.csv")  # Adjust path as needed
    # tuning results (these are tuned to extremes so redo for all if you want that)
    #file_list = ['/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_MW_EXTREMES.csv', 
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_NE_EXTREMES.csv',
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SE_EXTREMES.csv', 
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SW_EXTREMES.csv',
    #             '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_W_EXTREMES.csv']

    file_list = ['/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_MW_cv.csv', 
                '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_NE_cv.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SE_cv.csv', 
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_SW_cv.csv',
                 '/home/mfholth/subseasonal/weekly_data/rf_tuning/rf_tuning_results_W_cv.csv']
    
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
        
    tuning_df = pd.concat(df_list, axis=0)
    
    # Train models for 1-4 week-ahead forecasting
    for horizon in range(1, 5):  # 1 to 4 weeks ahead
        print(f"Running group PFI for {horizon}-week ahead forecasting...")
       
        for region in regions:
            print(f" - Processing region: {region}")
            row = tuning_df[(tuning_df['region'] == region) & (tuning_df['Horizon'] == horizon)]
    
            train_data[f"{region}_target"] = train_data[region].shift(-horizon)
            test_data[f"{region}_target"] = test_data[region].shift(-horizon)

            # only use features w/ positive importance 
            allowed_features = importance_df[
                (importance_df['Region'] == region) & (importance_df['Horizon'] == horizon)
            ]['Feature'].tolist()
            allowed_features = [feat for feat in allowed_features if feat not in redundant_cols]
    
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
        
            # Define features (include 'current week' as a predictor)
            lagged_features = [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, 6)]
            feature_cols = lagged_features + [f for f in allowed_features if f not in lagged_features]
    
            
            X_train = train_subset[feature_cols]
            y_train = train_subset[f"{region}_target"]
            X_test = test_subset[feature_cols]
            y_test = test_subset[f"{region}_target"]
    
    
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
            cluster_labels = cluster_df['Cluster']

            group_dict = {}
            for col, group_id in zip(X_test.columns, cluster_labels):
                group_dict.setdefault(group_id, []).append(col)

            group_importance_df = grouped_pfi(model, X_test, y_test, group_dict, metric=mean_squared_error, n_repeats=20)
            # Flatten the dictionary into a list of (group_id, variable) pairs
            group_data = [(group_id, var) for group_id, vars in group_dict.items() for var in vars]
            
            # Convert to DataFrame
            group_df = pd.DataFrame(group_data, columns=['Group', 'Feature'])
            
            # Optional: sort for easier inspection
            group_df = group_df.sort_values(by='Group').reset_index(drop=True)
            group_importance_df.to_csv(f'/home/mfholth/subseasonal/weekly_data/Grouped_PFI/GPFI_{region}_{horizon}_60.csv', index = False)
            group_df.to_csv(f'/home/mfholth/subseasonal/weekly_data/Grouped_PFI/Groups_{region}_{horizon}_60.csv', index = False)

            train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
            test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    
    # Print completion message
    print("All PFI done.")

if __name__ == "__main__":
    main()
    