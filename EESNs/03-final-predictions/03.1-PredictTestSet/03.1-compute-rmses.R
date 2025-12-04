########################################
########################################
## 03.1-compute-rmses.R               ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript to compute RMSEs           ##
## - Loads setup.R and functions.R    ##
## - Calculates RMSEs                 ##
## - Saves all RMSEs to results       ##
########################################
########################################


# SETUP & FUNCTIONS -------------------------------------------------------
## Load set up
source("../03.0-Setup/03-setup.R")

## Load functions
source("../03.0-Setup/03-functions.R")

## Load Initial FI Results
fi_initial <- fi_initial <- readr::read_csv(
  paste0(fp,
         "agu-manuscript-code/EESNs/02-grouped-feature-importance/results/gpfi-initial.csv"))

## Load the Best models
best_models_all <- read.csv('../../02-grouped-feature-importance/results/best-models-all.csv')
best_models_extreme <- read.csv('../../02-grouped-feature-importance/results/best-models-extreme.csv')

## Load Mean Temps that are used to compute residuals in observed data
mean_temps_1980_to_2020 <-
  read.csv(
    paste0(fp, "weekly_ave_test/CONUS_Regions_1980_to_2020.csv") 
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) 
mean_temps_2021_to_2024 <-
  read.csv(
    paste0(fp, "weekly_ave_test/CONUS_Regions_2021_to_2024.csv") 
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) 

mean_temps <- rbind(mean_temps_1980_to_2020, mean_temps_2021_to_2024)

## Clean up mean temps and lean up the mean temps and Zscores
temps <-
  mean_temps |>
  pivot_longer(cols = W:NE_Zscore, names_to = "region") |>
  separate(region, into = c("target_region", "stat"), fill = "right") |>
  mutate(stat = ifelse(is.na(stat), "temp_mean", stat)) |>
  pivot_wider(names_from = stat, values_from = value) |>
  rename(temp_zscore = Zscore)

## Clean and Combine Target Variables
obs <-
  bind_rows(
    outputs_train |> mutate(data = "training"),
    outputs_test |> mutate(data = "testing")
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) |>
  pivot_longer(cols = W:NE, names_to = "target_region", values_to = "obs")

## Join target variables and temps
obs_temps <-
  full_join(obs, temps, by = join_by(target_region, date, year, week)) |>
  mutate(extreme = ifelse(abs(temp_zscore) > 1, TRUE, FALSE))


# Compute RMSEs for Models optimized for All Data -------------------------
input_vars <- "Chosenby_AllData"
## Specify target regions
target_regions = c("MW", "NE", "SE", "SW", "W")

## Fit models and save predictions for each region

for (target_region in target_regions) {
  
  ## Current region 
  print(paste("Current target region:", target_region))
  
  ## Prepare data needed for computing RMSEs
  y_for_rmse <-
    obs_temps |> 
    filter(target_region == !!(target_region))
  
  ## Consider different forecast lags
  for (this_tau in taus) {
    
    ## Initial FI for this region and horizon
    fi_RH <- fi_initial |> 
      filter(tau == this_tau,
             region == target_region) |> 
      select(-region, -tau) 
    
    ## Specify m
    m <- fi_RH$m[1]
    
    ## Specify nu
    nu <- fi_RH$nu[1]/100
    
    ## Specify nh
    nh <- fi_RH$nh[1]
    
    ## Specify tau
    tau <- this_tau
    
    ## Find the number of groups used in the best model
    this_target_region <- target_region
    g <- (best_models_all |> 
            filter(tau == this_tau,
                   target_region == this_target_region,
                   data == 'training'))$best_model_all
  
    ## Status message
    print(paste0("Target Region:", target_region, " Tau:", tau, "   Using ", g, " out of ", max(fi_RH$rank), " groups"))
    
    ## Compute RMSE
    compute_rmses_subset(
      rmse_data =  y_for_rmse,
      y_region = target_region,
      x_regions = input_regions,
      x_vars = input_vars,
      tau = tau,
      m = m,
      nh = nh,
      nu = nu,
      nens = nens,
      groups = g
    )
    
  }
  
}


# List folder name where RMSEs live
rmse_folder <- paste0("../results/rmses-", input_vars)

# Get all names of RMSEs files
rmse_files = list.files(rmse_folder)

# Join RMSEs from all files
rmses <-
  set_names(rmse_files) |>
  map(
    .f = function(x)
      read.csv(paste0(rmse_folder, "/", x))
  ) |>
  list_rbind(names_to = "file")  
  

# Save RMSEs
write.csv(rmses, paste0("../results/", input_vars, "-rmses.csv"), row.names = FALSE)



# Compute RMSEs for Models optimized for Extremes -------------------------
input_vars <- "Chosenby_Extremes"
## Specify target regions
target_regions = c("MW", "NE", "SE", "SW", "W")

## Fit models and save predictions for each region

for (target_region in target_regions) {
  
  ## Current region 
  print(paste("Current target region:", target_region))
  
  ## Prepare data needed for computing RMSEs
  y_for_rmse <-
    obs_temps |> 
    filter(target_region == !!(target_region))
  
  ## Consider different forecast lags
  for (this_tau in taus) {
    
    ## Initial FI for this region and horizon
    fi_RH <- fi_initial |> 
      filter(tau == this_tau,
             region == target_region) |> 
      select(-region, -tau) 
    
    ## Specify m
    m <- fi_RH$m[1]
    
    ## Specify nu
    nu <- fi_RH$nu[1]/100
    
    ## Specify nh
    nh <- fi_RH$nh[1]
    
    ## Specify tau
    tau <- this_tau
    
    ## Find the number of groups used in the best model
    this_target_region <- target_region
    g <- (best_models_extreme |> 
            filter(tau == this_tau,
                   target_region == this_target_region,
                   data == 'training'))$best_model_extreme
    
    ## Status message
    print(paste0("Target Region:", target_region, " Tau:", tau, "   Using ", g, " out of ", max(fi_RH$rank), " groups"))
    
    ## Compute RMSE
    compute_rmses_subset(
      rmse_data =  y_for_rmse,
      y_region = target_region,
      x_regions = input_regions,
      x_vars = input_vars,
      tau = tau,
      m = m,
      nh = nh,
      nu = nu,
      nens = nens,
      groups = g
    )
    
  }
  
}


# List folder name where RMSEs live
rmse_folder <- paste0("../results/rmses-", input_vars)

# Get all names of RMSEs files
rmse_files = list.files(rmse_folder)

# Join RMSEs from all files
rmses <-
  set_names(rmse_files) |>
  map(
    .f = function(x)
      read.csv(paste0(rmse_folder, "/", x))
  ) |>
  list_rbind(names_to = "file") 
  

# Save RMSEs
write.csv(rmses, paste0("../results/", input_vars, "-rmses.csv"), row.names = FALSE)
