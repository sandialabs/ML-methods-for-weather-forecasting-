########################################
########################################
## 14.0-functions.R                   ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript containing functions       ##
## for fitting and using the ESN      ##
## - Fit the Ensemble ESN             ##
## - Predict using the Ensemble ESN   ##
## - Compute Predictive RMSE          ##
## ---------------------------------- ##
## Version 14.1                       ##
## This version focuses on using the  ##
## best model from version 13 and     ##
## using that model to predict the    ##
## final holdout years 2021-2024      ##
########################################
########################################

# FIT EESN ----------------------------------------------------------------
########################################
########################################
## Function to fit an ensemble ESN,   ##
## predict for the test set, then     ##
## save the predictions in a          ##
## results directory                  ##
########################################
########################################
# 
# y_train = targets_train
# y_test = targets_test
# y_region = target_region
# x_train = inputs_train
# x_test = inputs_test
# x_regions = input_regions
# x_vars = input_vars
# t_train = dates_train
# t_test = dates_test
# tau = tau
# m = m
# nh = nh
# nu = nu
# nens = nens
# seed = seed
# ncores = ncores

# Function for fitting EESN and saving predictions (for high and low extremes)
fit_regional_inputs_eesns <- 
  function(
    y_train,
    y_test,
    y_region,
    x_train,
    x_test,
    x_regions,
    x_vars,
    t_train,
    t_test,
    tau,
    m,
    nh,
    nu,
    nens,
    ncores,
    seed
  ) {
    
    ## Create file path for predictions for this unique set of tuning parameters
    dir.create(paste0("../results/14.1-preds-", x_vars), showWarnings = FALSE)
    fp_preds <-
      paste0(
        "../results/14.1-preds-", x_vars,
        "/targetregion", toupper(y_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau),
        "-m", ifelse(m < 10, paste0(0, m), m),
        "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
        "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
        "-nens", ifelse(nens < 100, paste0(0, nens), nens),
        ".csv"
      )
    
    ## Fit model and compute predictions for region - if not already saved
    if (!file.exists(fp_preds)) {
      
      # EESN
      eesn <-
        fit_Eesn(
          x = x_train |> as.matrix(),
          y = y_train |> as.matrix(),
          t = as.character(t_train),
          tau = tau,
          m = m,
          tau_emb = 1,
          nh = nh,
          nu = nu,
          n_ensembles = nens,
          cores = ncores,
          seed = seed
        )
      
      ## Compute predictions
      preds <-
        predict_eesn(
          eesn = eesn, 
          x_test = as.matrix(x_test),
          t_test = t_test
        )
      
      ## Save predictions
      write.csv(preds, fp_preds, row.names = FALSE)
      
    }
    
  }


# EESN PREDICTIONS --------------------------------------------------------
########################################
########################################
## Predict using a fitted EESN        ##
## Returns both in-sample and         ##
## out-of sample predictions (must    ##
## input test predictors and dates)   ##
########################################
########################################

## Function for getting predictions
predict_eesn <- function(eesn, x_test, t_test) {
  
  ## Get tau
  tau = eesn[[1]]$params_tuning$tau
  
  ## Training data predictions
  preds_train <-
    map(.x = eesn, .f = predict_esn) |>
    map(\(preds) data.frame(pred = preds$preds_ins) |> tibble::rownames_to_column(var = "date")) |>
    list_rbind(names_to = "ens_member") |>
    separate(date, into = c("year", "week"), sep = 4) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week)
    ) |>
    mutate(data = "training")
  
  ## Forecasts based on training data
  preds_oos <-
    map(.x = eesn, .f = predict_esn) |>
    map(\(preds) data.frame(pred = preds$preds_oos) |> tibble::rownames_to_column(var = "date")) |>
    list_rbind(names_to = "ens_member") |>
    separate(date, into = c('date', 'tau')) |>
    separate(date, into = c("year", "week"), sep = 4) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week),
      tau = as.numeric(tau)
    ) |>
    mutate(week = week + tau) |> 
    mutate(
      year = ifelse(week > 52, year + 1, year),
      new_week = ifelse(week > 52, week - 52, week)
    ) |>
    mutate(week = new_week) |>
    select(-new_week, -tau) |>
    mutate(data = "testing")
  
  ## Forecasts from testing data
  preds_test <-
    map(
      .x = eesn, 
      .f = predict_esn, 
      x_new = as.matrix(x_test),
      t_new = as.character(t_test)
    ) |>
    map(\(preds) data.frame(pred = preds$preds_new) |> tibble::rownames_to_column(var = "date")) |>
    list_rbind(names_to = "ens_member") |>
    separate(date, into = c('date', 'tau')) |>
    separate(date, into = c("year", "week"), sep = 4) |>
    mutate(
      year = as.numeric(year),
      week = as.numeric(week),
      tau = as.numeric(tau)
    ) |>
    mutate(week = week + tau) |> 
    mutate(
      year = ifelse(week > 52, year + 1, year),
      new_week = ifelse(week > 52, week - 52, week)
    ) |> 
    mutate(week = new_week) |>
    select(-new_week, -tau) |>
    mutate(data = "testing")
  
  ## Join and compute ensemble prediction
  preds <-
    bind_rows(preds_train, preds_oos, preds_test) |>
    na.omit() |> # remove dates where no predictions are given (due to lags and such)
    mutate(ens_member = paste0("pred", ens_member)) |>
    pivot_wider(names_from = "ens_member", values_from = "pred") |>
    rowwise() |>
    mutate(
      pred = mean(c_across(starts_with("pred")))
    ) |>
    ungroup() |>
    select(year, week, data, pred)
  
  ## Return the predictions
  return(preds)
  
}


# COMPUTE RMSE ------------------------------------------------------------
########################################
########################################
## Compute RMSE given a set of        ##
## predictions and actual output      ##
## Writes the RMSE results out to     ##
## a .csv file in the results dir     ##
########################################
########################################

## Function to compute RMSEs
compute_rmses <- function(
    rmse_data,  
    y_region,  
    x_regions,
    x_vars,
    tau,
    m,
    nh,
    nu,
    nens
) {
  
  ## Create general file path
  run_path <- 
    paste0(
      "/targetregion", toupper(y_region), 
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
      "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
      "-nens", ifelse(nens < 100, paste0(0, nens), nens),
      ".csv"
    )
  
  # Create file path for RMSEs
  dir.create(paste0("../results/14.1-rmses-", x_vars))
  fp_rmses = paste0("../results/14.1-rmses-", x_vars, run_path)
  
  # Create file path for predictions
  fp_preds = paste0("../results/14.1-preds-", x_vars, run_path)
  
  if (!file.exists(fp_rmses)) {
    if (file.exists(fp_preds)) {
      
      # Load predictions
      preds = read.csv(fp_preds)
      
      # Join predictions with observational data
      preds_obs <- 
        left_join(preds, rmse_data, by = join_by(year, week, data)) |>
        filter(year < 2020)
      
      # Compute RMSEs
      rmses <-
        preds_obs |>
        summarise(rmse = sqrt(sum((pred - obs)^2) / n()), .by = data)
      
      # Compute RMSEs with extreme temps
      rmses_extrems <-
        preds_obs |>
        filter(extreme) |>
        summarise(rmse_extreme = sqrt(sum((pred - obs)^2) / n()), .by = data)
      
      # Join RMSEs
      all_rmses = left_join(rmses, rmses_extrems, by = join_by(data))
      
      # Save RMSEs
      write.csv(all_rmses, fp_rmses, row.names = FALSE)
      
    }
  }
  
}



# FIT EESN while adding clusters -------------------------------------------------
########################################
########################################
## Function to fit an ensemble ESN,   ##
## predict for the test set, then     ##
## save the predictions in a          ##
## results directory                  ##
## Creates a subdirectory for each    ##
## region-horizon combination         ##
## You can add "message" which is a   ##
## string that will be appended onto  ##
## the end of the save file           ##
########################################
########################################
# 
# y_train = targets_train
# y_test = targets_test
# y_region = target_region
# x_train = inputs_train
# x_test = inputs_test
# x_regions = input_regions
# x_vars = input_vars
# t_train = dates_train
# t_test = dates_test
# tau = tau
# m = m
# nh = nh
# nu = nu
# nens = nens
# seed = seed
# ncores = ncores

# Function for fitting EESN and saving predictions (for high and low extremes)
fit_regional_inputs_eesns_subset <- 
  function(
    y_train,
    y_test,
    y_region,
    x_train,
    x_test,
    x_regions,
    x_vars,
    t_train,
    t_test,
    tau,
    m,
    nh,
    nu,
    nens,
    ncores,
    seed,
    groups
  ) {
    
    ## Create file path for predictions for this unique set of tuning parameters
    res_dir <- paste0("../results/preds-", x_vars)
    dir.create(res_dir, showWarnings = FALSE)
    fp_preds <-
      paste0(res_dir,
        "/targetregion", toupper(y_region),
        "-tau", ifelse(tau < 10, paste0(0, tau), tau),
        "-m", ifelse(m < 10, paste0(0, m), m),
        "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
        "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
        "-nens", ifelse(nens < 100, paste0(0, nens), nens),
        "-groups", groups,
        ".csv"
      )
    
    ## Fit model and compute predictions for region - if not already saved
    if (!file.exists(fp_preds)) {
      
      # EESN
      eesn <-
        fit_Eesn(
          x = x_train |> as.matrix(),
          y = y_train |> as.matrix(),
          t = as.character(t_train),
          tau = tau,
          m = m,
          tau_emb = 1,
          nh = nh,
          nu = nu,
          n_ensembles = nens,
          cores = ncores,
          seed = seed
        )
      
      ## Compute predictions
      preds <-
        predict_eesn(
          eesn = eesn, 
          x_test = as.matrix(x_test),
          t_test = t_test
        )
      
      ## Save predictions
      write.csv(preds, fp_preds, row.names = FALSE)
      
    }
    
  }




# COMPUTE RMSE WITH CLUSTERS ------------------------------------------------------------
########################################
########################################
## Compute RMSE given a set of        ##
## predictions and actual output      ##
## Writes the RMSE results out to     ##
## a .csv file in the results dir     ##
########################################
########################################

## Function to compute RMSEs
compute_rmses_subset <- function(
    rmse_data,  
    y_region,  
    x_regions,
    x_vars,
    tau,
    m,
    nh,
    nu,
    nens,
    groups
) {
  
  ## Create file path for predictions for this unique set of tuning parameters
  rmse_dir <- paste0("../results/rmses-", x_vars)
  dir.create(rmse_dir, showWarnings = FALSE)
  preds_dir <- paste0("../results/preds-", x_vars)
  
  
  run_path <- paste0(
           "/targetregion", toupper(y_region),
           "-tau", ifelse(tau < 10, paste0(0, tau), tau),
           "-m", ifelse(m < 10, paste0(0, m), m),
           "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
           "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
           "-nens", ifelse(nens < 100, paste0(0, nens), nens),
           "-groups", groups,
           ".csv"
    )
  
  fp_rmses <- paste0(rmse_dir, run_path)
  fp_preds = paste0(preds_dir, run_path)
  
  if (!file.exists(fp_rmses)) {
    if (file.exists(fp_preds)) {
      
      # Load predictions
      preds = read.csv(fp_preds)
      
      # Join predictions with observational data
      preds_obs <- 
        left_join(preds, rmse_data, by = join_by(year, week, data)) |>
        filter(year < 2025)
      
      # Compute RMSEs
      rmses <-
        preds_obs |>
        summarise(rmse = sqrt(sum((pred - obs)^2) / n()), .by = data)
      
      # Compute RMSEs with extreme temps
      rmses_extrems <-
        preds_obs |>
        filter(extreme) |>
        summarise(rmse_extreme = sqrt(sum((pred - obs)^2) / n()), .by = data)
      
      # Join RMSEs
      all_rmses = left_join(rmses, rmses_extrems, by = join_by(data))
      
      # Save RMSEs
      write.csv(all_rmses, fp_rmses, row.names = FALSE)
      
    }
  }
  
}
