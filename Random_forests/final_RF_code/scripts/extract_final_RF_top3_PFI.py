"""
Extract grouped-PFI information for the top groups used by the final RF models.

This does NOT retrain any RF models. It reads the iterative grouped-PFI model
selection results and the merged grouped-PFI output, applies the same percent-
tolerance rule used by run_final_RF_models_2021_2024.py, and writes a long-format
CSV with one row per feature in each of the top selected groups.

Run from final_code/scripts, e.g.:
    python extract_final_RF_top3_PFI.py --mode standard
    python extract_final_RF_top3_PFI.py --mode weighted
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PFI_DIR = PROJECT_ROOT / "outputs" / "feature_importance"


def select_percent_tolerance_row(
    results_df,
    region,
    horizon,
    rmse_col="Test_RMSE",
    pct_tolerance=0.01,
):
    sub_df = results_df[
        (results_df["Region"] == region)
        & (results_df["Horizon"] == horizon)
    ].copy()

    if sub_df.empty:
        raise ValueError(f"No iterative GPFI results for {region}, horizon {horizon}.")

    min_metric = sub_df[rmse_col].min()
    tolerance_threshold = min_metric * (1 + pct_tolerance)
    best_rows = sub_df[sub_df[rmse_col] <= tolerance_threshold]

    if best_rows.empty:
        raise ValueError(
            f"No model falls within {pct_tolerance:.1%} of the minimum "
            f"{rmse_col} for {region}, horizon {horizon}."
        )

    # Same rule used by get_best_features_with_percent_tolerance(): the first
    # model within tolerance is chosen. The iterative results are ordered from
    # fewer to more groups, so this is the smallest model within tolerance.
    selected_row = best_rows.iloc[0].copy()
    return selected_row, min_metric, tolerance_threshold


def extract_one_model(
    gpfi_df,
    selected_row,
    region,
    horizon,
    mode,
    rmse_col,
    pct_tolerance,
    min_metric,
    tolerance_threshold,
    top_n=3,
):
    selected_features = [
        f.strip()
        for f in str(selected_row["Feature_names"]).split(";")
        if f.strip()
    ]

    group_data = gpfi_df[
        (gpfi_df["Region"] == region)
        & (gpfi_df["Horizon"] == horizon)
        & (gpfi_df["Mean_Importance"] > 0)
    ].copy()

    if group_data.empty:
        raise ValueError(f"No positive grouped-PFI results for {region}, horizon {horizon}.")

    group_summary = (
        group_data.groupby("Group", as_index=False)
        .agg(
            Mean_Importance=("Mean_Importance", "mean"),
            Std_Importance=("Std_Importance", "mean"),
            Group_Size=("Group_Size", "first"),
        )
        .sort_values("Mean_Importance", ascending=False)
        .reset_index(drop=True)
    )

    selected_num_groups = int(selected_row["Num_Groups"])
    selected_group_summary = group_summary.iloc[:selected_num_groups].copy()

    # Ensure that the group reconstruction agrees exactly with the feature list
    # saved for the final percent-tolerance-selected model.
    selected_group_ids = set(selected_group_summary["Group"])
    reconstructed_features = set(
        group_data.loc[
            group_data["Group"].isin(selected_group_ids), "Feature"
        ].unique()
    )
    expected_features = set(selected_features)

    if reconstructed_features != expected_features:
        missing = sorted(expected_features - reconstructed_features)
        extra = sorted(reconstructed_features - expected_features)
        raise ValueError(
            f"PFI-group/feature mismatch for {region}, horizon {horizon}. "
            f"Missing from groups: {missing[:10]}; extra from groups: {extra[:10]}"
        )

    # Do not invent a third group if the final selected model contains only
    # one or two groups.
    n_to_save = min(top_n, selected_num_groups)
    top_groups = selected_group_summary.iloc[:n_to_save].copy()
    top_groups["Group_Rank"] = np.arange(1, n_to_save + 1)

    out = (
        group_data[["Group", "Feature"]]
        .drop_duplicates()
        .merge(top_groups, on="Group", how="inner", validate="many_to_one")
    )

    out["Region"] = region
    out["Horizon"] = horizon
    out["Mode"] = mode
    out["Selection_Metric"] = rmse_col
    out["Pct_Tolerance"] = pct_tolerance
    out["Minimum_Metric"] = min_metric
    out["Tolerance_Threshold"] = tolerance_threshold
    out["Selected_Model_Metric"] = selected_row[rmse_col]
    out["Selected_Num_Groups"] = selected_num_groups
    out["Selected_Num_Features"] = int(selected_row["Num_Features"])

    return out[
        [
            "Region",
            "Horizon",
            "Mode",
            "Selection_Metric",
            "Pct_Tolerance",
            "Minimum_Metric",
            "Tolerance_Threshold",
            "Selected_Model_Metric",
            "Selected_Num_Groups",
            "Selected_Num_Features",
            "Group_Rank",
            "Group",
            "Mean_Importance",
            "Std_Importance",
            "Group_Size",
            "Feature",
        ]
    ].sort_values(["Region", "Horizon", "Group_Rank", "Feature"])


def main(mode="standard", pct_tolerance=0.01, top_n=3):
    if mode == "standard":
        rmse_col = "Test_RMSE"
    elif mode == "weighted":
        rmse_col = "Weighted_RMSE_smooth"
    else:
        rmse_col = "Test_RMSE_extreme"

    iterative_file = PFI_DIR / f"iterative_group_rmse_results_RF_TS_{mode}.csv"
    gpfi_file = PFI_DIR / f"All_GPFI_Merged_RF_TS_{mode}.csv"

    iterative_df = pd.read_csv(iterative_file)
    gpfi_df = pd.read_csv(gpfi_file)

    rows = []
    for region in sorted(iterative_df["Region"].unique()):
        for horizon in sorted(iterative_df.loc[iterative_df["Region"] == region, "Horizon"].unique()):
            selected_row, min_metric, tolerance_threshold = select_percent_tolerance_row(
                iterative_df,
                region,
                horizon,
                rmse_col=rmse_col,
                pct_tolerance=pct_tolerance,
            )

            rows.append(
                extract_one_model(
                    gpfi_df,
                    selected_row,
                    region,
                    horizon,
                    mode,
                    rmse_col,
                    pct_tolerance,
                    min_metric,
                    tolerance_threshold,
                    top_n=top_n,
                )
            )

    out_df = pd.concat(rows, ignore_index=True)
    out_file = PFI_DIR / f"RF_final_top{top_n}_grouped_PFI_{mode}.csv"
    out_df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print(f"Rows: {len(out_df)}")
    print("Groups saved per model:")
    print(
        out_df[["Region", "Horizon", "Group_Rank", "Group"]]
        .drop_duplicates()
        .groupby(["Region", "Horizon"])
        .size()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standard", "extreme", "weighted"],
        default="standard",
    )
    parser.add_argument(
        "--pct-tolerance",
        type=float,
        default=0.01,
        help="Fractional tolerance around the minimum metric; 0.01 = 1 percent.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Maximum number of top selected PFI groups to save per model.",
    )
    args = parser.parse_args()
    main(args.mode, args.pct_tolerance, args.top_n)
