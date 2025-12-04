# Load set up
source("setup.R")

# Load functions
source("functions.R") # CEE

# Fit EESNs for one CV fold
run_eesn_tuning_for_one_cv_split(
  cv_splits = cv_splits,
  split = 3,
  target_region = "W",
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