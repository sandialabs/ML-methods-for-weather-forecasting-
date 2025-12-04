# Load set up
source("local-setup.R")

# Load functions
source("local-functions.R") # CEE

# Fit EESNs for one CV fold
run_eesn_tuning_for_one_cv_split(
  cv_splits = cv_splits,
  split = 5,
  target_region = "MW",
  targets = outputs_train,
  inputs = inputs_train,
  taus = taus,
  ms = ms,
  nhs = nhs,
  nus = nus,
  nens = nens,
  seed = seed,
  ncores = ncores
)
