## Rscript to fit final EESNs
## Uses 1980-2020 as training data with the goal of predicting the final 
## holdout years 2021-2024

# SETUP -------------------------------------------------------------------

# Specify data type
data = "test"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")

# Specify target region
this_target_region = "MW"

# Load Initial FI Results
fi_initial = readr::read_csv('../results/gpfi-initial.csv')


# Best Models Based on All Test Data and Extremes -------------------------

# Load the best models
best_models_all = read.csv("../results/best-subset-models-all.csv")
best_models_extreme = read.csv('../results/best-subset-models-extreme.csv')
best_models_weighted = read.csv('../results/best-subset-models-weighted.csv')

# Extract target variables
targets_train_val = outputs_train_val |> select(all_of(this_target_region))
colnames(targets_train_val) = paste0(this_target_region, "_current_week")
targets_test = outputs_test |> select(all_of(this_target_region))
colnames(targets_test) = paste0(this_target_region, "_current_week")

# Extract dates
dates_train_val = sort(unique(outputs_train_val$date))
dates_test = sort(unique(outputs_test$date))

# Separate inputs into train/test
inputs_train_val <-
  inputs_train_val_test |> 
  filter(year < test_year) |>
  select(-year, -week, -date) |>
  bind_cols(targets_train_val)
colnames(inputs_train_val) <-
  str_remove_all(colnames(inputs_train_val), "_weekave")
inputs_test <-
  inputs_train_val_test |> 
  filter(year >= test_year) |>
  select(-year, -week, -date) |>
  bind_cols(targets_test)
colnames(inputs_test) <-
  str_remove_all(colnames(inputs_test), "_weekave")

for (this_tau in taus) {
  
  # Status Message
  print(paste0("Target Region:", this_target_region, " Tau:", this_tau))
  
  # FI for this region and horizon
  fi_RH <- 
    fi_initial |> 
    filter(
      tau == this_tau,
      region == this_target_region
    ) |> 
    select(-region, -tau)
  
  # Specify m
  m = fi_RH$m[1]
  
  # Specify number of layers
  nlayers = fi_RH$nlayers[1]
  
  # Specify nu
  nu = fi_RH$nu[1] / 100
  
  # Specify nh
  nh = fi_RH$nh[1]
  
  # Find the number of groups used in the best models
  g_all <- 
    (best_models_all |> 
       filter(
         tau == this_tau,
         target_region == this_target_region,
         data == 'training'
        )
    )$best_model_all
  g_extreme <- 
    (best_models_extreme |> 
       filter(
         tau == this_tau,
         target_region == this_target_region,
         data == 'training')
     )$best_model_extreme
  g_weighted <- 
    (best_models_weighted |> 
       filter(
         tau == this_tau,
         target_region == this_target_region,
         data == 'training')
    )$best_model_weighted
  
  # List of variables from a subset of groups
  remove_vars_all = (fi_RH |> filter(rank > g_all))$var
  remove_vars_extreme = (fi_RH |> filter(rank > g_extreme))$var
  remove_vars_weighted = (fi_RH |> filter(rank > g_weighted))$var
  
  # Only include variables in the subset
  inputs_train_val_subset_all <- 
    inputs_train_val |>
    select(-all_of(remove_vars_all))
  inputs_test_subset_all <- 
    inputs_test |>
    select(-all_of(remove_vars_all))
  inputs_train_val_subset_extreme <-
    inputs_train_val |>
    select(-all_of(remove_vars_extreme))
  inputs_test_subset_extreme <- 
    inputs_test |>
    select(-all_of(remove_vars_extreme))
  inputs_train_val_subset_weighted <-
    inputs_train_val |>
    select(-all_of(remove_vars_weighted))
  inputs_test_subset_weighted <- 
    inputs_test |>
    select(-all_of(remove_vars_weighted))
  
  # Fit models and save predictions
  fit_regional_eesn(
    y_train = targets_train_val,
    y_test = targets_test,
    y_region = this_target_region,
    x_train = inputs_train_val_subset_all,
    x_test = inputs_test_subset_all,
    t_train = dates_train_val,
    t_test = dates_test,
    tau = this_tau,
    m = m,
    nlayers = nlayers,
    nh = nh,
    nu = nu,
    nens = nens,
    ncores = ncores,
    seed = seed,
    fp_res = fp_res,
    folder_name = "final-preds-chosen-by-all-data",
    message = paste0("groups", g_all), 
    show_var = TRUE
  )
  fit_regional_eesn(
    y_train = targets_train_val,
    y_test = targets_test,
    y_region = this_target_region,
    x_train = inputs_train_val_subset_extreme,
    x_test = inputs_test_subset_extreme,
    t_train = dates_train_val,
    t_test = dates_test,
    tau = this_tau,
    m = m,
    nlayers = nlayers,
    nh = nh,
    nu = nu,
    nens = nens,
    ncores = ncores,
    seed = seed,
    fp_res = fp_res,
    folder_name = "final-preds-chosen-by-extremes",
    message = paste0("groups", g_extreme),
    show_var = TRUE
  )
  fit_regional_eesn(
    y_train = targets_train_val,
    y_test = targets_test,
    y_region = this_target_region,
    x_train = inputs_train_val_subset_weighted,
    x_test = inputs_test_subset_weighted,
    t_train = dates_train_val,
    t_test = dates_test,
    tau = this_tau,
    m = m,
    nlayers = nlayers,
    nh = nh,
    nu = nu,
    nens = nens,
    ncores = ncores,
    seed = seed,
    fp_res = fp_res,
    folder_name = "final-preds-chosen-by-weighted",
    message = paste0("groups", g_weighted),
    show_var = TRUE
  )
  
}

