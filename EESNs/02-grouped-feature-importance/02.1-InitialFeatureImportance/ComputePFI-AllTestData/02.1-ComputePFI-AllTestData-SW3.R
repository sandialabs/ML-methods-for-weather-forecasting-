########################################
########################################
## 02.1-ComputePFI-AllTestData-SW3.R  ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript to calculate PFI for the   ## 
## groups for clusters of related     ##
## input variables. This script will  ##
## compute the grouped permutation    ##
## feature importance on the best     ## 
## model for each region-horizon      ## 
## combination as identified from the ## 
## tuning done in 01                  ##
## - Loads the best rmses             ##
## - Fits an EESN for SW3             ##
## - Calculates PFI for each group    ##
########################################
########################################


# Script Options ----------------------------------------------------------
this_target_region <- 'SW'
this_tau <- 3


# SETUP -------------------------------------------------------------------
## Load set up
source("../../02.0-Setup/02-setup.R")
source("../../02.0-Setup/02-compute_gfi_series.R")


# DATA STEPS --------------------------------------------------------------
## Load the best performing models
best_rmses <- read.csv("../../../01-tuning-time-series-cv/results/cv-rmses-best.csv")

## Read in clusters from Maike
## (Remove the other lagged variables because of how we are using m)
`%ni%` <- Negate(`%in%`)

clusters <-  readr::read_csv("../All_GPFI_Merged.csv") |> 
  select(Group, Feature, Region, Horizon) |> 
  mutate(middle = substr(Feature, 3, 6)) |> 
  filter(middle %ni% c("_lag", "lag_")) |> 
  select(-middle)


# FEATURE IMPORTANCE ------------------------------------------------------
## Number of reps for feature importance
nreps <-  1

## Prepare Directory Path
fi_dir <- paste0(
  "../../results/GPFI",
  "-nreps", ifelse(nreps < 10, paste0(0, nreps), nreps), "/")
dir.create(fi_dir)

## Fit model and save Feature Importance
## Prepare tuning params
this_model <- best_rmses |> 
  filter(target_region == this_target_region,
         tau == this_tau)

target_region = this_model[1,]$target_region
tau = this_model[1,]$tau
m = this_model[1,]$m
nh = this_model[1,]$nh
nu = this_model[1,]$nu


## Subset Clusters 
clusters_RH <- clusters |> 
  filter(Region == target_region,
         Horizon == tau)

## Prepare file path
fi_file <- 
  paste0(fi_dir,
         "targetregion", target_region, 
         "-tau", ifelse(tau < 10, paste0(0, tau), tau),
         "-m", ifelse(m < 10, paste0(0, m), m),
         "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
         "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
         ".csv")

if (!file.exists(fi_file)) {
  
  ## Extract target variables
  targets_train = outputs_train |> select(all_of(target_region))
  colnames(targets_train) <- paste0(target_region, "_current_week")
  targets_test = outputs_test |> select(all_of(target_region))
  colnames(targets_test) <- paste0(target_region, "_current_week")
  
  ## Separate inputs into train/test and convert to matrices
  inputs_train <-
    climate_inputs |> 
    filter(year < test_year) |>
    select(-year, -week, -date) |>
    bind_cols(targets_train)
  
  inputs_test <-
    climate_inputs |> 
    filter(year >= test_year) |>
    select(-year, -week, -date) |>
    bind_cols(targets_test)
  
  ## Fit EESN
  eesn <-
    fit_Eesn(
      x = inputs_train |> as.matrix(),
      y = targets_train |> as.matrix(),
      t = as.character(dates_train),
      tau = tau,
      m = m,
      tau_emb = 1,
      nh = nh,
      nu = nu,
      n_ensembles = nens,
      cores = ncores,
      seed = seed
    )
  
  ## Compute FI
  fi <-
    compute_gfi_series(
      model = eesn,
      nreps = nreps, 
      clusters_RH = clusters_RH)
  
  # Save FI
  write.csv(
    x = data.frame(fi),
    file = fi_file,
    row.names = FALSE
  ) 
  
}
