import pandas as pd
import numpy as np
import os
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from collections import defaultdict
from subseasonal.config import REDUNDANT_COLS, PFI_DIR, RAW_DIR, REGIONS, HORIZONS, MAX_LAG
from subseasonal.GPFI import load_train_test_data_dev


importance_df = pd.read_csv(PFI_DIR / "feature_importance.csv")  
train_data, _ = load_train_test_data_dev()

# Loop over all region-horizon combinations
for region in REGIONS:
    for horizon in HORIZONS:
        # Subset the data
        train_data[f"{region}_target"] = train_data[region].shift(-horizon)

        # this is just for getting all possible features for every region/horizon combo
        allowed_features = importance_df[
            (importance_df['Region'] == region) & (importance_df['Horizon'] == horizon)
        ]['Feature'].tolist()
        allowed_features = [feat for feat in allowed_features if feat not in REDUNDANT_COLS]

        # Create 'current week' predictor (target variable at lag 0)
        train_data[f"{region}_current_week"] = train_data[region]  # No shift

        # Create up to 5 lagged predictors
        for lag in range(1, 6):  # Lags 1 to 5
            train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)


        lagged_features = [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, MAX_LAG)]
        feature_cols = lagged_features + [f for f in allowed_features if f not in lagged_features]
        train_subset = train_data[feature_cols]
       
        # Drop rows with NaN targets (last few weeks in training set)
        X_train = train_subset.dropna()
  
        # Compute correlation matrix
        corr_matrix = X_train.corr(method='pearson')  
        train_data, _ = load_train_test_data_dev()

        # Save to CSV
        filename = Path(PFI_DIR / f"correlation_matrices/corr_{region}_h{horizon}.csv"
        corr_matrix.to_csv(filename)
        print(f"Saved: {filename}")