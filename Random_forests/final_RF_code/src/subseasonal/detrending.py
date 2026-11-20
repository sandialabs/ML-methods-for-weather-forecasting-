# src/detrending.py
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

from .config import REGIONS, TIME_FEATURES, REGION_ZSCORES, SPLITS, PROJECT_ROOT, DETREND_DIR
from .io import read_csv_with_date, ensure_dir, save_dataframe
from .metrics import weighted_rmse_step, weighted_rmse_smooth

@dataclass
class RegionDetrendResult:
    region: str
    model: LinearRegression
    train_rmse: float
    test_rmse: float


def load_detrending_data(split_key: str) -> pd.DataFrame:
    spec = SPLITS[split_key]
    print(f"using data for {split_key}", flush=True)
    weekly = read_csv_with_date(spec["data_file"])
    time_inputs = read_csv_with_date(spec["time_file"])
    data = pd.merge(weekly, time_inputs, on="Date").set_index("Date")
    return data


def split_train_test(data: pd.DataFrame, train_end_year: int):
    years = data.index.str[:4].astype(int)
    train = data.loc[years <= train_end_year].copy()
    test = data.loc[years > train_end_year].copy()
    return train, test


def fit_linear_models_by_region(train_data, test_data):
    results = {}
    models = {}
    # Loop through each region
    for region, zscore_col in zip(REGIONS, REGION_ZSCORES):
        print(f"Processing region: {region}")
    
        # Define predictors and target
        X_train = train_data[TIME_FEATURES]
        y_train = train_data[region]
        X_test = test_data[TIME_FEATURES]
        y_test = test_data[region]
    
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        models[region] = {
        'Region': region,
        'model': model
        }
    
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        # Compute RMSE for test set
        rmse_all = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
        # Compute RMSE for "extremes" (Z-score < -1 or Z-score > 1)
        extreme_mask = (test_data[zscore_col] < -1) | (test_data[zscore_col] > 1)
        if extreme_mask.sum() > 0:  # Avoid errors if no extremes exist
            rmse_extreme = np.sqrt(mean_squared_error(y_test[extreme_mask], y_test_pred[extreme_mask]))
        else:
            rmse_extreme = np.nan  # Assign NaN if no extremes exist

        # Convert to numpy arrays and remove any potential NaNs jointly
        z_data = test_data[zscore_col]
        valid_mask = (~pd.isna(y_test.values)) & (~pd.isna(y_test_pred)) & (~pd.isna(z_data.values))
            
        y_true_valid = y_test.values[valid_mask]
        y_pred_valid = y_test_pred[valid_mask]
        z_valid = z_data.values[valid_mask]

        # compute weighted scores ala Taggart 2022 (this is NOT a proper score!)
        wrmse_step = weighted_rmse_step(y_true_valid, y_pred_valid, z_valid)
        wrmse_smooth = weighted_rmse_smooth(y_true_valid, y_pred_valid, z_valid, alpha=1.0)
        # Store model and errors
        results[region] = {
            'model': model,
            'rmse_all': rmse_all,
            'rmse_extreme': rmse_extreme,
            'Weighted_RMSE_step': wrmse_step,
            'Weighted_RMSE_smooth': wrmse_smooth,
            'residuals_test': y_test - y_test_pred,
            'residuals_train': y_train - y_train_pred,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
        print(f"RMSE (All): {rmse_all:.4f}, RMSE (Extremes): {rmse_extreme:.4f}")

    metrics = pd.DataFrame.from_dict({r: {'RMSE_All': v['rmse_all'], 'RMSE_Extreme': v['rmse_extreme'], 'Weighted_RMSE_step': v['Weighted_RMSE_step'], 'Weighted_RMSE_smooth': v['Weighted_RMSE_smooth']} for r, v in results.items()}, orient='index')
    test_residuals = pd.DataFrame({region: results[region]['residuals_test'] for region in REGIONS})
    train_residuals = pd.DataFrame({region: results[region]['residuals_train'] for region in REGIONS})

    test_preds = pd.DataFrame({region: results[region]['y_test_pred'] for region in REGIONS})
    train_preds = pd.DataFrame({region: results[region]['y_train_pred'] for region in REGIONS})
    return metrics, test_residuals, train_residuals, test_preds, train_preds, models


def save_models(models: dict, out_dir: Path):
    ensure_dir(out_dir)
    for region, model in models.items():
        joblib.dump(model, out_dir / f"{region}_linear_model.pkl")


def save_detrending_outputs(
    split_key: str,
    models: dict,
    train_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    test_residuals: pd.DataFrame,
    metrics: pd.DataFrame,
):
    spec = SPLITS[split_key]
    out_dir = Path(DETREND_DIR / spec["label"])
    ensure_dir(out_dir)

    test_preds['Date'] = test_residuals.index
    test_preds.set_index('Date', inplace=True)
    
    train_preds['Date'] = train_residuals.index
    train_preds.set_index('Date', inplace=True)


    save_dataframe(train_preds, out_dir / "train_preds.csv", index=True)
    save_dataframe(test_preds, out_dir / "test_preds.csv", index=True)
    save_dataframe(train_residuals, out_dir / "train_residuals.csv", index=True)
    save_dataframe(test_residuals, out_dir / "test_residuals.csv", index=True)
    save_dataframe(metrics, out_dir / "test_rmse_results.csv", index=True)
    save_models(models, out_dir / "models")

