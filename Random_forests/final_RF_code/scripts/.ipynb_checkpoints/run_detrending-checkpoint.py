# scripts/run_detrending.py
from src.config import DATASETS
from src.detrending import (
    load_detrending_data,
    split_train_test,
    fit_linear_models_by_region,
    save_detrending_outputs,
)

def main(split_key="dev"):
    spec = SPLITS[split_key]

    data = load_data(spec)
    train_df, test_df = split_train_test(data, spec["train_end_year"])

    results = fit_linear_models_by_region(train_df, test_df)

    save_detrending_outputs(
        dataset_key=dataset_key,
        models=models,
        train_preds=train_preds,
        test_preds=test_preds,
        train_residuals=train_residuals,
        test_residuals=test_residuals,
        metrics_df=metrics_df,
    )

if __name__ == "__main__":
    main()
