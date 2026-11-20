#### SETUP ####

# Specify data type
data = "train"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")

# Create folder for results
fp_res_rmses = paste0(fp_res, "cv-rmses/")
if (!file.exists(fp_res_rmses)) dir.create(fp_res_rmses, showWarnings = FALSE)

#### DATA STEPS ####

# Subset training data outputs to those used in the times series CV and 
# pivot data to long format (for use later)
outputs_train_long <- 
  outputs_train |>
  #slice(cv_splits$val_start[4]:cv_splits$val_end[n_splits]) |>
  pivot_longer(
    cols = W:NE,
    names_to = "target_region", 
    values_to = "obs"
  )

# Load mean temps that are used to compute residuals in observed data
mean_temps <-
  read.csv(paste0(fp_data, "weekly_ave_test/CONUS_Regions_1980_to_2020.csv")) |>
  rename(date = Date) |>
  mutate(date = as.character(date)) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  )

# Clean up the mean temps and zscores
mean_temps_clean <-
  mean_temps |>
  pivot_longer(cols = W:NE_Zscore, names_to = "region") |>
  separate(region, into = c("target_region", "stat"), fill = "right") |>
  mutate(stat = ifelse(is.na(stat), "temp_mean", stat)) |>
  pivot_wider(names_from = stat, values_from = value) |>
  rename(temp_zscore = Zscore)

# Join target variables and temps (via left join so only CV dates used for
# validation folds are kept) and determine extreme dates
obs_temps <-
  left_join(
    outputs_train_long, 
    mean_temps_clean, 
    by = join_by(target_region, date, year, week)
  ) |>
  mutate(extreme = ifelse(abs(temp_zscore) > 1, TRUE, FALSE))

#### RMSES ####

# Specify target regions
target_regions = c("MW", "NE", "SE", "SW", "W")

# Fit models and save predictions for each region
for (target_region in target_regions) {
  
  # Current region 
  print(paste("Current target region:", target_region))
  
  # Prepare data needed for computing RMSEs
  y_obs_for_rmse <-
    obs_temps |> 
    filter(target_region == !!(target_region))
  
  # Consider different forecast lags
  for (tau in taus) {
    # Consider different values of m
    for (m in ms) {
      # Consider different number of layers
      for (nlayer in nlayers) {
        # Consider different numbers of hidden units
        for (nh in nhs) {
          # Consider different values of nu 
          for (nu in nus) {
            # Fit model and save predictions
            compute_cv_rmses(
              y_obs = y_obs_for_rmse,
              y_region = target_region,
              n_splits = n_splits,
              tau = tau,
              m = m,
              nlayers = nlayer,
              nh = nh,
              nu = nu,
              fp_res = fp_res
            )
          }
        }
      }
    }
  }
  
}

# Get all names of RMSEs files
rmse_folder = paste0(fp_res, "cv-rmses/")
rmse_files = list.files(rmse_folder)

# Join RMSEs from all files
rmses_by_cv_split <-
  set_names(rmse_files) |>
  map(
    .f = function(x)
      read.csv(paste0(rmse_folder, "/", x))
  ) |>
  list_rbind(names_to = "file")

# Compute RMSEs across all 5 folds when only 1 layers
rmses_single_layer_all_folds <-
  rmses_by_cv_split |>
  filter(nlayers == 1) |>
  summarise(
    rmse = mean(rmse),
    rmse_extreme = mean(rmse_extreme),
    rmse_weighted = mean(rmse_weighted),
    .by = file
  ) |>
  mutate(folds = "all")

# Compute RMSEs across folds 4 and 5 for 1-3 layers
rmses_folds_4_and_5 <-
  rmses_by_cv_split |>
  filter(cv_val_split %in% 4:5) |>
  summarise(
    rmse = mean(rmse),
    rmse_extreme = mean(rmse_extreme),
    rmse_weighted = mean(rmse_weighted),
    .by = file
  ) |>
  mutate(folds = "4_and_5")

# Join RMSEs from all files
rmses <-
  bind_rows(
    rmses_single_layer_all_folds,
    rmses_folds_4_and_5
  )

# Save RMSEs
write.csv(
  rmses_by_cv_split, 
  paste0(fp_res, "cv-rmses-by-split.csv"), 
  row.names = FALSE
)
write.csv(
  rmses,
  paste0(fp_res, "cv-rmses.csv"),
  row.names = FALSE
)
