########################################
########################################
## 14.0-setup.R                       ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript to set up ESN models       ##
## - Loads Packages                   ##
## - Loads Data                       ##
## - Specifies Model/Tuning Options   ##
## ---------------------------------- ##
## Version 14.1                       ##
## This version focuses on using the  ##
## best model from version 13 and     ##
## using that model to predict the    ##
## final holdout years 2021-2024      ##
########################################
########################################

# PACKAGES ----------------------------------------------------------------
## Load packages
library(dplyr)
library(listenr)
library(lubridate)
library(purrr)
library(stringr)
library(tidyr)

# DATA STEPS --------------------------------------------------------------
## Path to subseasonal_extreme folder

### Path for working locally
local_fp = "//cee/projects/subseasonal_extreme/"
## Path for working on the CEE Compute Server
cee_fp = "/projects/subseasonal_extreme/"

## Select which path to use
fp =  cee_fp

## Variables identified by Maike as redundant
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

## Input Climate Data
climate_inputs <-
  readr::read_csv(
    paste0(fp, "weekly_ave_test/weekly_aves_regional_inputs_1980_2024.csv"), 
    show_col_types = FALSE
  ) |>
  mutate(date = as.character(date)) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) |>
  select(-all_of(redundant_cols))

## Training Data Outputs
outputs_train <-
  read.csv(
    paste0(fp, "weekly_ave_test/Results/weekly_model_results_2021-2024/train_residuals.csv")
  )

## Testing Data Outputs
outputs_test <-
  read.csv(
    paste0(fp, "weekly_ave_test/Results/weekly_model_results_2021-2024/test_residuals.csv")
  )

# MODEL OPTIONS -----------------------------------------------------------

## Year when testing data starts
test_year = 2021

## Forecast lags to use
taus = 1:4

## Which input regions to consider 
input_regions = "all"

## Which input variables to consider (label for saving)
input_vars = "FinalTestSet"

## Values of m to consider
ms = c(0, 1, 2, 3, 4)

## Number of hidden units to consider
nhs = c(50, 100, 250, 500, 1000, 2000)

## Values of nu to consider
nus = c(0.10, 0.35, 0.60, 0.85)

## Number of ensembles
nens = 25

## Number of cores to be used for fitting
ncores = 25  ## ncores = nens is often a good idea if possible

## Seed
seed = 20250219

## Extract dates
dates_train = sort(unique(outputs_train$Date))
dates_test = sort(unique(outputs_test$Date))
