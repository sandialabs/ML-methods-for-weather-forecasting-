## Rscript to fit EESN                ##
## - Iteratively adds groups of input ##
##   variables in the order of their  ##
##   initial feature importance.      ##

# Script Options ----------------------------------------------------------

this_target_region = 'SW'
this_tau = 2


# SETUP -------------------------------------------------------------------

# Specify data type
data = "val"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")

# Load Initial FI Results
fi_initial = readr::read_csv("../results/gpfi-initial.csv")


# MODELS ------------------------------------------------------------------

# Specify target region
target_region = this_target_region

# Specify tau
tau = this_tau

# FI for this region and horizon
fi_RH <-
  fi_initial |> 
  filter(
    tau == this_tau,
    region == target_region
  ) |> 
  select(-region, -tau) 

# Specify m
m = fi_RH$m[1]

# Specify m
nlayers = fi_RH$nlayers[1]

# Specify nu
nu = fi_RH$nu[1] / 100

# Specify nh
nh = fi_RH$nh[1]

# Extract target variables
targets_train = outputs_train |> select(all_of(target_region))
colnames(targets_train) = paste0(target_region, "_current_week")
targets_val = outputs_val |> select(all_of(target_region))
colnames(targets_val) = paste0(target_region, "_current_week")

# Separate inputs into train/val and convert to matrices
inputs_train <-
  inputs_train_val |> 
  filter(year < val_year) |>
  select(-year, -week, -date) |>
  bind_cols(targets_train)
colnames(inputs_train) = str_remove_all(colnames(inputs_train), "_weekave")

inputs_val <-
  inputs_train_val |> 
  filter(year >= val_year) |>
  select(-year, -week, -date) |>
  bind_cols(targets_val)
colnames(inputs_val) = str_remove_all(colnames(inputs_val), "_weekave")

# Start with a single cluster of input variables then add one cluster at a time
# Fit models and save predictions for each number of input groups
pb = txtProgressBar(min=0, max=max(fi_RH$rank), style=3)
l = 0
for(g in rev(1:max(fi_RH$rank))){
  
  # Status message
  print(paste0(
    "Target Region:", target_region, 
    " Tau:", tau, 
    "   Using ", g, 
    " out of ", max(fi_RH$rank), 
    " groups"
  ))
  
  # List of variables from a subset of groups
  remove_vars = (fi_RH |> filter(rank > g))$var
  
  # Only include variables in the subset
  inputs_train_subset <- 
    inputs_train |>
    select(-all_of(remove_vars))
  inputs_val_subset <- 
    inputs_val |>
    select(-all_of(remove_vars))
  
  # Extract dates
  dates_train = sort(unique(outputs_train$date))
  dates_val = sort(unique(outputs_val$date))
  
  # Fit model and save predictions
  fit_regional_eesn(
    y_train = targets_train,
    y_test = targets_val,
    y_region = target_region,
    x_train = inputs_train_subset,
    x_test = inputs_val_subset,
    t_train = dates_train,
    t_test = dates_val,
    tau = tau,
    m = m,
    nlayers = nlayers,
    nh = nh,
    nu = nu,
    nens = nens,
    ncores = ncores,
    seed = seed,
    fp_res = "../results/",
    folder_name = "best-subset-preds",
    message = paste0("groups", g)
  )
  
  # Move Progress Bar
  l = l+1
  setTxtProgressBar(pb, l)
  
}
