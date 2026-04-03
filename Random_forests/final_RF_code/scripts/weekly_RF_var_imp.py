import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from pathlib import path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import SPLITS, HORIZONS, REGIONS, PFI_DIR
from subseasonal.io import read_csv_with_date
from subseasonal.GPFI import load_train_test_data_dev

######################################################################
# this script is only for calculating permutation feature importance
# not used for paper
# only used in downstream scripts to easily grab inputs for all models
######################################################################


def main():
    # Load input features
    
    train_data, test_data = load_train_test_data_dev()
    
    
    # Dictionary to store trained models and results
    models = {}
    rmse_results = []
    feature_importances = []
    
    # Train models for 1-4 week-ahead forecasting
    for horizon in HORIZONS:  # 1 to 4 weeks ahead
        print(f"Training models for {horizon}-week ahead forecasting...")
       
        for region in REGIONS:
            print(f" - Processing region: {region}")
    
            X_train, y_train = build_XY_data(train_data, region, horizon, importance_df)

            X_test, y_test = build_XY_data(test_data, region, horizon, importance_df)
    
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
    
    # Save feature importances
    feature_importance_df = pd.DataFrame(feature_importances)
    feature_importance_df.to_csv(Path(PFI_DIR / "feature_importance.csv"), index=False))

    # Print completion message
    print("done.")

if __name__ == "__main__":
    main()