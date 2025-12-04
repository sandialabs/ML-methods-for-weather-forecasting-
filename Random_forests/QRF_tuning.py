import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss


def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: python qrf_tuning.py <region> <quantile>")

    region = sys.argv[1]
    quantile = float(sys.argv[2])

    valid_regions = ["SW", "SE", "NE", "MW", "W"]
    if region not in valid_regions:
        raise ValueError(f"Invalid region '{region}'. Must be one of: {', '.join(valid_regions)}")
    if not (0 < quantile < 1):
        raise ValueError("Quantile must be a float between 0 and 1 (exclusive).")

    regional_inputs = pd.read_csv("/home/mfholth/subseasonal/weekly_data/weekly_aves_regional_inputs_1980_2020.csv")
    regional_inputs.rename({'date': 'Date'}, axis=1, inplace=True)
    train_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/train_residuals.csv")
    test_residuals = pd.read_csv("/home/mfholth/subseasonal/weekly_data/results/test_residuals.csv")

    for df in [regional_inputs, train_residuals, test_residuals]:
        df['Date'] = df['Date'].astype(str)

    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")

    os.makedirs("/home/mfholth/subseasonal/weekly_data/qrf_tuning/", exist_ok=True)

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [5, 10, 20, 50, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
    }

    tuning_results = []

    for horizon in range(1, 5):
        print(f"Tuning QRF for {region} - {horizon}-week ahead, quantile={quantile}", flush=True)

        train_data[f"{region}_target"] = train_data[region].shift(-horizon)
        test_data[f"{region}_target"] = test_data[region].shift(-horizon)

        train_data[f"{region}_current_week"] = train_data[region]
        test_data[f"{region}_current_week"] = test_data[region]

        for lag in range(1, 6):
            train_data[f"{region}_lag_{lag}"] = train_data[region].shift(lag)
            test_data[f"{region}_lag_{lag}"] = test_data[region].shift(lag)

        all_cols = [f"{region}_target"] + [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, 6)]
        all_cols += [col for col in regional_inputs.columns if col != "Date"]

        train_subset = train_data.dropna()
        feature_cols = [f"{region}_current_week"] + [f"{region}_lag_{l}" for l in range(1, 6)]
        feature_cols += [col for col in regional_inputs.columns if col != "Date"]

        X_train = train_subset[feature_cols]
        y_train = train_subset[f"{region}_target"]


        def pinball_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return -mean_pinball_loss(y, y_pred, alpha=quantile)

        qrf = RandomForestQuantileRegressor(random_state=42, q=quantile)

        grid_search = GridSearchCV(qrf, param_grid, cv=4, scoring=pinball_scorer, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_

        result = {
            "Horizon": horizon,
            "Best Pinball Loss": best_score,
            "region": region,
            "quantile": quantile
        }
        result.update(best_params)
        tuning_results.append(result)

        print(f"   - Best Pinball Loss: {best_score:.4f}", flush=True)
        print(f"   - Best Parameters: {best_params}", flush=True)

    tuning_results_df = pd.DataFrame(tuning_results)
    tuning_results_df.to_csv(f"/home/mfholth/subseasonal/weekly_data/qrf_tuning/qrf_tuning_results_{region}_q{int(quantile*100)}.csv", index=False)
    print("\n✅ QRF hyperparameter tuning complete. Results saved.", flush=True)


if __name__ == "__main__":
    main()