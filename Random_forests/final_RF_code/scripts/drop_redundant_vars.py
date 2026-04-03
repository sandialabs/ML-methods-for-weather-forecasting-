import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
# Path setup for Slurm/HPC compatibility
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from subseasonal.GPFI import load_train_test_data



def remove_collinear_features(df_model, target_var, threshold, verbose=False):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold and which have the least correlation with the target (dependent) variable. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        df_model: features dataframe
        target_var: target (dependent) variable
        threshold: features with correlations greater than this value are removed
        verbose: set to "True" for the log printing

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    datanp = df_model.to_numpy()
    corr_matrix = np.corrcoef(datanp, rowvar=False)
    #iters = range(len(df_model) - 1)
    iters=range(df_model.shape[1] - 1)
    drop_cols = []
    dropped_feature = ""

    corrmat_df = pd.DataFrame(corr_matrix, index = df_model.columns, columns = df_model.columns)
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        if i % 500 ==0:
            print(i)
        for j in range(i+1):
            item = corrmat_df.iloc[j:(j+1), (i+1):(i+2)] 
            col = item.columns
            row = item.index
            val = abs(item.values[0][0])

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = df_model[col.values[0]].corr(target_var)
                row_value_corr = df_model[row.values[0]].corr(target_var)
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-------------------------------------------------")

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df_model = df_model.drop(columns=drops)

    print("dropped columns: ")
    print(list(drops))
    print("----------------------------------------------")
    print("used columns: ")
    print(df_model.columns.tolist())

    return df_model

def main(region="W", horizon=1):
    train_data, _ = load_train_test_data_dev()
    train_data = train_data.copy()
    train_data[f"{region}_target"] = data[region].shift(-horizon)
    
    train_data_subset = train_data.drop[REGIONS]

    target_var = [f"{region}_target"]
    print(f"target var is {target_var}")
    
    remove_collinear_features(train_data_subset, target_var, threshold=.95, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        choices=["W", "SW", "MW", "SE", "NE"],
        default="W",
        help="Which region for the target var",
    )
    parser.add_argument("--horizon", 
                        choices=[1, 2, 3, 4],
                        default=1,
                        help="Which horizon for the target var",
    )
    args = parser.parse_args()
    main(args.region, args.horizon)
