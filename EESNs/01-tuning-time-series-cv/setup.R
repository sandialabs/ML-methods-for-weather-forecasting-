#### SETUP ####

# Install the correct version of listenr (if not already installed)
# install.packages(
#   "../../../listenr-versions/listenr-v0.4.1.tar.gz",
#   repos = NULL,
#   type="source"
# )

# Load packages
library(dplyr)
library(listenr) # using v0.4.1
library(lubridate)
library(purrr)
library(stringr)
library(tidyr)

# Specify file path
fp = "../../../" # CEE

#### DATA ####

# Load inputs and extract training data
inputs_train <-
  readr::read_csv(
    paste0(fp, "analysis-data/weekly_aves_regional_inputs_1980_2020.csv"),
    show_col_types = FALSE
  ) |>
  mutate(date = as.character(date)) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) |> 
  filter(year < 2017)

# Load training data outputs
outputs_train <-
  readr::read_csv(
    paste0(fp, "weekly_ave_test/Results/weekly_model_results/train_residuals.csv"),
    show_col_types = FALSE
  ) |>
  rename(date = Date) |>
  mutate(date = as.character(date)) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  )

#### EESN PARAMETERS/SETTINGS ####

# Forecast lag
taus = 1:4

# Specify m values to consider
ms = c(0, 1, 2, 3, 4)

# Specify number of hidden units to consider
nhs = c(50, 100, 250, 500, 1000)

# Specify values of nu to consider
nus = c(0.10, 0.35, 0.60, 0.85)

# Number of cores to be used for fitting
nens = 25

# Number of cores to use
ncores = 25

# Seed
seed = 20250219

#### CV FOLDS ####

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
