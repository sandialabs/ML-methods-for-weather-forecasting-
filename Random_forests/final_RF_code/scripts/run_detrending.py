# scripts/run_detrending.py
import sys
from pathlib import Path
import argparse
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from subseasonal.config import SPLITS

from subseasonal.detrending import (
    load_detrending_data,
    split_train_test,
    fit_linear_models_by_region,
    save_detrending_outputs,
)

def main(split_key="dev"):
    spec = SPLITS[split_key]

    data = load_detrending_data(split_key)
    train_data, test_data = split_train_test(data, spec["train_end_year"])

    metrics, test_residuals, train_residuals, test_preds, train_preds, models = fit_linear_models_by_region(train_data, test_data)

    save_detrending_outputs(
        split_key=split_key,
        models=models,
        train_preds=train_preds,
        test_preds=test_preds,
        train_residuals=train_residuals,
        test_residuals=test_residuals,
        metrics=metrics,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "final"],
        default="dev",
        help="Which train/test split to run",
    )
    args = parser.parse_args()
    main(args.split)

