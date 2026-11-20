## Rscript to compute RMSEs
## - Loads setup.R and functions.R
## - Calculates RMSEs
## - Saves all RMSEs to results


# SETUP & FUNCTIONS -------------------------------------------------------

# Specify data type
data = "val"

# Load set up
source("../setup.R")

# Load functions
source("../functions.R")

# Load initial feature importance results
fi_initial = readr::read_csv('../results/gpfi-initial.csv')

# Load Mean Temps that are used to compute residuals in observed data
mean_temps <-
  read.csv(
    paste0(fp_data, "weekly_ave_test/CONUS_Regions_1980_to_2020.csv") 
  ) |>
  rename(date = Date) |>
  separate(date, into = c("year", "week"), sep = 4, remove = FALSE) |>
  mutate(
    year = as.numeric(year),
    week = as.numeric(week)
  ) 

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
    outputs_train |> mutate(data = "training"),
    outputs_val |> mutate(data = "testing")
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

    for(i in 1:max(fi_RH$rank)){
      
      # Status message
      print(paste0(
        "Target Region:", target_region, 
        " Tau:", tau, 
        "   Using ", i, 
        " out of ", max(fi_RH$rank), 
        " groups"
      ))
      
      # Compute RMSE
      compute_rmses_subset(
        rmse_data = y_for_rmse,
        y_region = target_region,
        tau = tau,
        m = m,
        nlayers = nlayers,
        nh = nh,
        nu = nu,
        nens = nens,
        groups = i,
        cut_off_year = 2020,
        folder_rmses = "best-subset-rmses",
        folder_preds = "best-subset-preds",
        fp_res = fp_res
      )
      
    }
    
    # Join RMSE Results by Region and Tau
    rmse_folder <- 
      paste0(
        "../results/best-subset-rmses",
        "/targetregion", toupper(target_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau)
      )
    
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
    write.csv(
      rmses, 
      paste0(rmse_folder, "-rmses.csv"), 
      row.names = FALSE
    )
    
  }
  
}

