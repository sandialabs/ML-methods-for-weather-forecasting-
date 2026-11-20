## Rscript to compute RMSEs
## - Loads setup.R and functions.R
## - Calculates RMSEs
## - Saves all RMSEs to results


# SETUP & FUNCTIONS -------------------------------------------------------

# Specify data type
data = "test"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")


# DATA STEPS --------------------------------------------------------------

# Load Initial FI Results
fi_initial = readr::read_csv('../results/gpfi-initial.csv')

# Load the best models
best_models_all = read.csv("../results/best-subset-models-all.csv")
best_models_extreme = read.csv('../results/best-subset-models-extreme.csv')
best_models_weighted = read.csv('../results/best-subset-models-weighted.csv')

# Load Mean Temps that are used to compute residuals in observed data
mean_temps_1980_to_2020 <-
  read.csv(
    paste0(fp_data, "weekly_ave_test/CONUS_Regions_1980_to_2020.csv") 
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) 
mean_temps_2021_to_2024 <-
  read.csv(
    paste0(fp_data, "weekly_ave_test/CONUS_Regions_2021_to_2024.csv") 
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) 
mean_temps <- rbind(mean_temps_1980_to_2020, mean_temps_2021_to_2024)

# Clean up mean temps and lean up the mean temps and Zscores
temps <-
  mean_temps |>
  pivot_longer(cols = W:NE_Zscore, names_to = "region") |>
  separate(region, into = c("target_region", "stat"), fill = "right") |>
  mutate(stat = ifelse(is.na(stat), "temp_mean", stat)) |>
  pivot_wider(names_from = stat, values_from = value) |>
  rename(temp_zscore = Zscore) |>
  mutate(date = as.character(date))

# Clean and Combine Target Variables
obs <-
  bind_rows(
    outputs_train_val |> mutate(data = "training"),
    outputs_test |> mutate(data = "testing")
  ) |>
  pivot_longer(cols = W:NE, names_to = "target_region", values_to = "obs")

# Join target variables and temps
obs_temps <-
  full_join(obs, temps, by = join_by(target_region, date, year, week)) |>
  mutate(extreme = ifelse(abs(temp_zscore) > 1, TRUE, FALSE))


# Compute RMSEs -----------------------------------------------------------

# Specify target regions
target_regions = c("MW", "NE", "SE", "SW", "W")

# Fit models and save predictions for each region
for (target_region in target_regions) {
  
  # Current region 
  print(paste("Current target region:", target_region))
  
  # Prepare data needed for computing RMSEs
  y_for_rmse <-
    obs_temps |> 
    filter(target_region == !!(target_region))
  
  # Consider different forecast lags
  for (this_tau in taus) {
    
    # Initial FI for this region and horizon
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
    nu = fi_RH$nu[1]/100
    
    # Specify nh
    nh = fi_RH$nh[1]
    
    # Specify tau
    tau = this_tau
    
    # Find the number of groups used in the best model
    this_target_region = target_region
    g_all <- (
      best_models_all |> 
        filter(
          tau == this_tau,
          target_region == this_target_region,
          data == 'training'
        )
      )$best_model_all
    g_extreme <- (
      best_models_extreme |> 
        filter(
          tau == this_tau,
          target_region == this_target_region,
          data == 'training'
        )
      )$best_model_extreme
    g_weighted <- (
      best_models_weighted |> 
        filter(
          tau == this_tau,
          target_region == this_target_region,
          data == 'training'
        )
    )$best_model_weighted
    
    # Status message
    print(paste0(
      "Target Region:", target_region, 
      " Tau:", tau, 
      "   Using ", g_all, 
      " out of ", max(fi_RH$rank), 
      " groups (all)"
    ))
    
    # Compute RMSE (all data)
    compute_rmses_subset(
      rmse_data = y_for_rmse,
      y_region = target_region,
      tau = tau,
      m = m,
      nlayers = nlayers,
      nh = nh,
      nu = nu,
      nens = nens,
      groups = g_all,
      cut_off_year = 2025,
      fp_res = fp_res,
      folder_rmses = "final-rmses-chosen-by-all-data",
      folder_preds = "final-preds-chosen-by-all-data"
    )
    
    # Status message
    print(paste0(
      "Target Region:", target_region, 
      " Tau:", tau, 
      "   Using ", g_extreme, 
      " out of ", max(fi_RH$rank), 
      " groups (extreme)"
    ))
    
    # Compute RMSE (extremes)
    compute_rmses_subset(
      rmse_data =  y_for_rmse,
      y_region = target_region,
      tau = tau,
      m = m,
      nlayers = nlayers,
      nh = nh,
      nu = nu,
      nens = nens,
      groups = g_extreme,
      cut_off_year = 2025,
      fp_res = fp_res,
      folder_rmses = "final-rmses-chosen-by-extremes",
      folder_preds = "final-preds-chosen-by-extremes"
    )
    
    # Status message
    print(paste0(
      "Target Region:", target_region, 
      " Tau:", tau, 
      "   Using ", g_weighted, 
      " out of ", max(fi_RH$rank), 
      " groups (weighted)"
    ))
    
    # Compute RMSE (extremes)
    compute_rmses_subset(
      rmse_data =  y_for_rmse,
      y_region = target_region,
      tau = tau,
      m = m,
      nlayers = nlayers,
      nh = nh,
      nu = nu,
      nens = nens,
      groups = g_weighted,
      cut_off_year = 2025,
      fp_res = fp_res,
      folder_rmses = "final-rmses-chosen-by-weighted",
      folder_preds = "final-preds-chosen-by-weighted"
    )
    
    # Join RMSE Results by Region and Tau
    rmse_folder_all <- 
      paste0(
        "../results/final-rmses-chosen-by-all-data",
        "/targetregion", toupper(target_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau)
      )
    rmse_folder_extreme <- 
      paste0(
        "../results/final-rmses-chosen-by-extremes",
        "/targetregion", toupper(target_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau)
      )
    rmse_folder_weighted <- 
      paste0(
        "../results/final-rmses-chosen-by-weighted",
        "/targetregion", toupper(target_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau)
      )
    
    # Get all names of RMSEs files
    rmse_files_all = list.files(rmse_folder_all)
    rmse_files_extreme = list.files(rmse_folder_extreme)
    rmse_files_weighted = list.files(rmse_folder_weighted)
    
    # Join RMSEs from all files
    rmses_all <-
      set_names(rmse_files_all) |>
      map(
        .f = function(x)
          read.csv(paste0(rmse_folder_all, "/", x))
      ) |>
      list_rbind(names_to = "file")
    rmses_extreme <-
      set_names(rmse_files_extreme) |>
      map(
        .f = function(x)
          read.csv(paste0(rmse_folder_extreme, "/", x))
      ) |>
      list_rbind(names_to = "file")
    rmses_weighted <-
      set_names(rmse_files_weighted) |>
      map(
        .f = function(x)
          read.csv(paste0(rmse_folder_weighted, "/", x))
      ) |>
      list_rbind(names_to = "file")
    
    # Save RMSEs
    write.csv(
      rmses_all, 
      paste0(rmse_folder_all, "-rmses.csv"), 
      row.names = FALSE
    )
    write.csv(
      rmses_extreme, 
      paste0(rmse_folder_extreme, "-rmses.csv"), 
      row.names = FALSE
    )
    write.csv(
      rmses_weighted, 
      paste0(rmse_folder_weighted, "-rmses.csv"), 
      row.names = FALSE
    )
    
  }
  
}
