# Specify data type
data = "train"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")

# Fit EESNs for one CV fold
for (curr_split in 4:5) {
  run_eesn_tuning_for_one_cv_split(
    cv_splits = cv_splits,
    split = curr_split,
    target_region = "SE",
    targets = outputs_train,
    inputs = inputs_train_val,
    taus = taus,
    ms = ms,
    nlayers = 3,
    nhs = nhs,
    nus = nus,
    nens = nens,
    seed = seed,
    ncores = ncores,
    fp_res = fp_res
  )
}