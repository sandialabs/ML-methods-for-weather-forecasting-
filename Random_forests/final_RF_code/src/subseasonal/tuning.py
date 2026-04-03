# /src/subseasonal/tuning.py
from .io import read_csv_with_date, ensure_dir
from .config import SPLITS, PROJECT_ROOT, DETREND_OUTPUTS, TUNING_DIR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path
import pandas as pd
import numpy as np

def load_dev_training_data() -> tuple[pd.DataFrame, list[str]]:
    spec = SPLITS["dev"]
    resid_spec = DETREND_OUTPUTS["dev"]
    regional_inputs = read_csv_with_date(spec["inputs_file"])
    
    input_feature_cols = regional_inputs.columns.tolist()
    train_residuals = read_csv_with_date(resid_spec["train_residuals_file"])
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    return train_data, input_feature_cols

def extreme_scoring_function(estimator, X, y, zscore_target):
    """
    Custom scorer for GridSearchCV.

    X.index is the reference_date t.
    y is the residual target at forecast_date t+h.
    zscore_target is a Series indexed by reference_date t, where values are
    the TRUE z-scores at forecast_date t+h.

    Returns negative RMSE over only the extreme cases, because GridSearchCV
    maximizes the score.
    """
    y_pred = estimator.predict(X)

    # Align z-scores to the rows being scored
    z = zscore_target.loc[X.index]
    mask = z.abs() > 1

    # If a fold has no extremes, return a large negative penalty
    if mask.sum() == 0:
        return -1e6

    return -mean_squared_error(y[mask], y_pred[mask])


def run_rf_grid_search(param_grid, X_train, y_train, scoring = "neg_mean_squared_error"):
    tscv = TimeSeriesSplit(n_splits=4)
    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    grid_search = GridSearchCV(rf, 
                                   param_grid,
                                   cv=tscv, 
                                   scoring=scoring,
                                   n_jobs=-1, 
                                   verbose=1)
        
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)
    return best_params, best_score

def save_tuning_results(tuning_results, region, type="standard"):
    tuning_results_df = pd.DataFrame(tuning_results)
    ensure_dir(TUNING_DIR / type)
    out_dir = TUNING_DIR / type
    tuning_results_df.to_csv(out_dir / f"rf_tuning_results_{region}_tscv_{type}.csv", index=False)
    

# functions for filling in tuning params for RF
def fill_or_default(val, default):
    return default if pd.isna(val) else val

def parse_maxdepth(val):
    if pd.isna(val):
        return None
    return int(val)
