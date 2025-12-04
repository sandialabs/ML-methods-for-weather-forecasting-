#### EESN TUNING ####

# cv_splits = cv_splits
# split = 6
# target_region = "MW"
# targets = outputs_train
# inputs = inputs_train
# taus = taus
# ms = ms
# nhs = nhs
# nus = nus
# nens = nens
# seed = seed
# ncores = ncores

# Function for fitting EESNs for one time series CV split over a set of 
# specified tuning parameter values
run_eesn_tuning_for_one_cv_split <- function(
    cv_splits,
    split,
    target_region,
    targets,
    inputs,
    taus,
    ms,
    nhs,
    nus,
    nens,
    seed,
    ncores
  ) {
  
  # Prepare the CV splits ends/starts
  train_start = cv_splits$train_start[split]
  train_end = cv_splits$train_end[split]
  val_start = cv_splits$val_start[split]
  val_end = cv_splits$val_end[split]
  
  # Separate target variable into CV folds
  target_train = targets |> select(all_of(target_region))
  target_train_folds = target_train |> slice(train_start:train_end)
  target_val_folds = target_train |> slice(val_start:val_end)
  
  # Separate inputs into CV folds and convert to matrices for listenr
  inputs_train_folds <-
    inputs |> 
    slice(train_start:train_end) |>
    select(-year, -week, -date) |>
    bind_cols(target_train_folds)
  inputs_val_folds <-
    inputs |> 
    slice(val_start:val_end) |>
    select(-year, -week, -date) |>
    bind_cols(target_val_folds)
  
  # Determine dates associated with CV folds
  dates_train_folds = inputs$date[train_start:train_end]
  dates_val_folds = inputs$date[val_start:val_end]
  
  # Consider different forecast lags
  for (tau in taus) {
    # Current tau
    print(paste("Current tau:", tau))
    # Consider different values of m
    for (m in ms) {
      # Consider different numbers of hidden units
      for (nh in nhs) {
        # Consider different values of nu 
        for (nu in nus) {
          # Fit model and save predictions
          fit_regional_eesn(
            y_region = target_region,
            y_train = target_train_folds,
            y_test = target_val_folds,
            x_train = inputs_train_folds,
            x_test = inputs_val_folds,
            t_train = dates_train_folds,
            t_test = dates_val_folds,
            split = split,
            tau = tau,
            m = m,
            nh = nh,
            nu = nu,
            nens = nens,
            seed = seed,
            ncores = ncores
          )
        }
      }
    }
  }
  
}

#### FIT REGIONAL EESN ####

# y_region = target_region
# y_train = target_train_folds
# y_test = target_val_folds
# x_train = inputs_train_folds
# x_test = inputs_val_folds
# t_train = dates_train_folds
# t_test = dates_val_folds
# split = split
# tau = taus[1]
# m = ms[1]
# nh = nh[1]
# nu = nu[1]
# nens = nens
# seed = seed
# ncores = ncores

# Function for fitting EESN and saving predictions
fit_regional_eesn <- 
  function(
    y_train,
    y_test,
    y_region,
    x_train,
    x_test,
    t_train,
    t_test,
    split = split,
    tau,
    m,
    nh,
    nu,
    nens,
    ncores,
    seed
  ) {
    
  # Create file path for predictions
  fp_preds <-
    paste0(
      "results/preds/",
      "targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nh", ifelse(
        nh < 100, 
        paste0("00", nh),
        ifelse(nh < 1000, paste0(0, nh), nh)
      ),
      "-nu", ifelse(
        nu == 0, 
        "000", 
        ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))
      ),
      "-split", split,
      ".csv"
    )
    
  # Fit model and compute predictions for region - if not already saved
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
    
    # Compute predictions
    preds <-
      predict_eesn(
        eesn = eesn, 
        x_test = as.matrix(x_test),
        t_test = t_test
      )
    
    # Save predictions
    write.csv(preds, fp_preds, row.names = FALSE)
    
  }
    
}

#### EESN PREDICTIONS ####

# eesn = eesn
# x_test = as.matrix(x_test)
# t_test = t_test

# Function for getting predictions
predict_eesn <- function(eesn, x_test, t_test) {
  
  # Get tau
  tau = eesn[[1]]$params_tuning$tau
  
  # Training data predictions
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
  
  # Forecasts based on training data
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
    mutate(data = "validation")
  
  # Forecasts from testing data
  preds_val <-
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
    mutate(data = "validation")
  
  # Join and compute ensemble prediction
  preds <-
    bind_rows(preds_train, preds_oos, preds_val) |>
    na.omit() |> # remove dates where no predictions are given (due to lags and such)
    mutate(ens_member = paste0("pred", ens_member)) |>
    pivot_wider(names_from = "ens_member", values_from = "pred") |>
    rowwise() |>
    mutate(
      pred = mean(c_across(starts_with("pred")))
    ) |>
    ungroup() |>
    select(year, week, data, pred)
  
  # Return the predictions
  return(preds)
  
}

#### COMPUTE RMSES ####

# y_obs = y_obs_for_rmse
# y_region = target_region
# n_splits = n_splits
# tau = taus[1]
# m = ms[1]
# nh = nhs[1]
# nu = nus[1]

compute_cv_rmses <- function(
    y_obs,
    y_region,
    n_splits,
    tau,
    m,
    nh,
    nu
) {

  # Create general file path
  fp_preds_rmses <-
    paste0(
      "targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nh", ifelse(
        nh < 100,
        paste0("00", nh),
        ifelse(nh < 1000, paste0(0, nh), nh)
      ),
      "-nu", ifelse(
        nu == 0,
        "000",
        ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))
      )
    )

  # Create file path for RMSEs
  fp_rmses = paste0("results/rmses/", fp_preds_rmses, ".csv")

  if (!file.exists(fp_rmses)) {

    # Load and join CV predictions on validation folds
    cv_val_preds <- 
      map(
        .x = 1:n_splits, 
        .f = function(split) {
          preds <-
            read.csv(paste0(
              "results/preds/", 
              fp_preds_rmses, 
              "-split", 
              split, 
              ".csv"
            )) |>
            filter(data == "validation") |>
            # remove last tau since (technically next CV fold)
            filter(row_number() <= n() - tau) |> 
            mutate(cv_val_split = split)
      }) |> 
      list_rbind()
    
    # Join predictions with observational data
    preds_obs = left_join(y_obs, cv_val_preds, by = join_by(year, week))

    # Compute RMSEs
    rmses <-
      preds_obs |>
      summarise(
        rmse = sqrt(sum((pred - obs)^2) / n()), 
        .by = cv_val_split
      )

    # Compute RMSEs with extreme temps
    rmses_extrems <-
      preds_obs |>
      filter(extreme) |>
      summarise(
        rmse_extreme = sqrt(sum((pred - obs)^2) / n()), 
        .by = cv_val_split
      )

    # Join RMSEs
    all_rmses = left_join(rmses, rmses_extrems, by = join_by(cv_val_split))

    # Save RMSEs
    write.csv(all_rmses, fp_rmses, row.names = FALSE)

  }

}
