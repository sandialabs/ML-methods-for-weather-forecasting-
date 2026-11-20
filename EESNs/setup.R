## Rscript to set up EESN models
## - Loads Packages
## - Loads Data
## - Specifies Model/Tuning Options

# PACKAGES ----------------------------------------------------------------

# Install the correct version of listenr (if not already installed) using 
# one of the following options
# remotes::install_github("sandialabs/listenr@v0.5.0")
# install.packages(
#   "../../../listenr-versions/listenr-v0.5.0.tar.gz",
#   repos = NULL,
#   type="source"
# )

# Load packages
library(dplyr)
library(listenr) # using v0.5.0
library(lubridate)
library(purrr)
library(stringr)
library(tidyr)

# DATA STEPS --------------------------------------------------------------

# Path to subseasonal_extreme folder (in CEE space)
fp_data = "../../../"

# Path to results older (in CEE space)
fp_res = "../results/"

# Create a results folder (if not already created)
if (!file.exists(fp_res)) dir.create(fp_res, showWarnings = FALSE)

# Year when validation data starts
val_year = 2017

# Year when testing data starts
test_year = 2021

# Variables identified by Maike as redundant
redundant_cols <-
  c(
    'slp_weekave_atlantic_ocean_mean',
    'h_850_weekave_conus_mean',
    'h_850_weekave_atlantic_ocean_pc4',
    'h_850_weekave_mexico_gulf_mean',
    'slp_weekave_mexico_gulf_mean',
    'slp_weekave_conus_pc1',
    'ts_weekave_conus_pc1',
    'slp_weekave_atlantic_trop_mean',
    'ts_weekave_southern_canada_pc1',
    't_850_weekave_southern_canada_mean',
    'h_850_weekave_atlantic_ocean_pc5',
    'slp_weekave_atlantic_ocean_pc6',
    'h_500_weekave_pacific_trop_mean',
    'h_850_weekave_atlantic_trop_mean',
    't_500_weekave_pacific_trop_mean',
    't_500_weekave_atlantic_trop_pc1',
    'h_850_weekave_pacific_ocean_pc1',
    't_850_weekave_conus_pc1',
    'slp_weekave_atlantic_ocean_pc1',
    'slp_weekave_pacific_trop_mean',
    't_850_weekave_southern_canada_pc1',
    'h_850_weekave_arctic_pc1',
    'h_500_weekave_atlantic_trop_mean',
    'h_850_weekave_atlantic_ocean_pc2',
    'slp_weekave_atlantic_ocean_pc3',
    'h_850_weekave_atlantic_ocean_pc7',
    't_500_weekave_mexico_gulf_mean',
    't_500_weekave_southern_canada_pc1',
    'slp_weekave_pacific_trop_pc2',
    'h_850_weekave_arctic_mean',
    'ts_weekave_atlantic_trop_mean',
    'h_850_weekave_pacific_trop_mean'
  )

if (data == "train") {
  
  # Load training + validation data inputs
  inputs_train_val <-
    readr::read_csv(
      paste0(fp_data, "analysis-data/weekly_aves_regional_inputs_1980_2020.csv"),
      show_col_types = FALSE
    ) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    ) |> 
    filter(year < val_year)
  
  # Load training data outputs (1980 through 2016)
  outputs_train <-
    readr::read_csv(paste0(
      fp_data, 
      "weekly_ave_test/Results/weekly_model_results/train_residuals.csv"
    )) |>
    rename(date = Date) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    )
  
} else if (data == "val") {
  
  # Load training + validation data inputs
  inputs_train_val <-
    readr::read_csv(
      paste0(
        fp_data, 
        "analysis-data/weekly_aves_regional_inputs_1980_2020.csv"
      ),
      show_col_types = FALSE
    ) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    ) |> 
    select(-all_of(redundant_cols))
  
  # Load training data outputs (1980 through 2016)
  outputs_train <-
    readr::read_csv(paste0(
      fp_data,
      "weekly_ave_test/Results/weekly_model_results/train_residuals.csv"
    )) |>
    rename(date = Date) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    )
  
  # Load validation data outputs (2017 through 2020)
  outputs_val <-
    read.csv(paste0(
      fp_data,
      "weekly_ave_test/Results/weekly_model_results/test_residuals.csv"
    )) |>
    rename(date = Date) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    )
  
} else if (data == "test") {
  
  # Load testing data inputs
  inputs_train_val_test <-
    readr::read_csv(
      paste0(
        fp_data, 
        "weekly_ave_test/weekly_aves_regional_inputs_1980_2024.csv"
      ), 
      show_col_types = FALSE
    ) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    ) |>
    select(-all_of(redundant_cols))
  
  # Load training and validation data outputs (1980 through 2020)
  outputs_train_val <-
    read.csv(paste0(
      fp_data,
      "weekly_ave_test/Results/weekly_model_results_2021-2024/Detrending_outputs/train_residuals.csv"
    )) |>
    rename(date = Date) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    )
  
  # Load testing data outputs (2021 through 2024)
  outputs_test <-
    read.csv(paste0(
      fp_data,
      "weekly_ave_test/Results/weekly_model_results_2021-2024/Detrending_outputs/test_residuals.csv"
    )) |>
    rename(date = Date) |>
    mutate(date = as.character(date)) |>
    separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    )
  
} else {
  
  print("Data type not specified")

}

# MODEL SETTINGS ----------------------------------------------------------

# Forecast lag
taus = 1:4

# Specify m values to consider
ms = c(0, 1, 2, 3, 4)

# Specify number of layers to consider
nlayers = 1:3

# Specify number of hidden units to consider
nhs = c(50, 100, 250, 500, 1000)

# Specify values of nu to consider
nus = c(0.10, 0.35, 0.60, 0.85)

# Number of cores to be used for fitting
nens = 25

# Number of cores to use
ncores = 25 ## ncores = nens is often a good idea if possible

# Seed
seed = 20250219

# CV FOLDS ----------------------------------------------------------------

if (data == "train") {
  
  # Training data size
  n = nrow(outputs_train)
  
  # Specify the number of splits
  n_splits = 5
  
  # Prepare the CV folds
  cv_splits <-
    data.frame(split = 1:(n_splits)) |> 
    mutate(
      train_size = split * floor(n / (n_splits + 1)) + (n %% (n_splits + 1)),
      val_size = floor(n / (n_splits + 1))
    ) |> 
    mutate(
      train_start = 1,
      train_end = train_size,
      val_start = train_size + 1, 
      val_end = train_size + val_size
    )
  
}
