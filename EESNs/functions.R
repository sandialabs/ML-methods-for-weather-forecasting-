## Rscript containing functions for fitting and using the ESN
## - Fit the Ensemble ESN
## - Predict using the Ensemble ESN
## - Compute Predictive RMSE

# EESN TIME SERIES CV TUNING ----------------------------------------------

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
    nlayers,
    nhs,
    nus,
    nens,
    seed,
    ncores,
    fp_res
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
            y_train = target_train_folds,
            y_test = target_val_folds,
            y_region = target_region,
            x_train = inputs_train_folds,
            x_test = inputs_val_folds,
            t_train = dates_train_folds,
            t_test = dates_val_folds,
            tau = tau,
            m = m,
            nlayers = nlayers,
            nh = nh,
            nu = nu,
            nens = nens,
            ncores = ncores,
            seed = seed,
            fp_res = fp_res,
            folder_name = "cv-preds",
            message = paste0("split", split)
          )
        }
      }
    }
  }

}

# FIT EESNS ---------------------------------------------------------------

# Function to fit EESN and save predictions.
# Predicts for the test set and saves the predictions in a results directory.
# Creates a subdirectory for each region-horizon combination. You can add
# "message" which is a string that will be appended onto the end of the
# save file.
fit_regional_eesn <-
  function(
    y_train,
    y_test,
    y_region,
    x_train,
    x_test,
    t_train,
    t_test,
    tau,
    m,
    nlayers,
    nh,
    nu,
    nens,
    ncores,
    seed,
    fp_res,
    folder_name,
    message, 
    show_var = FALSE
) {

  # Create file path for CV predictions
  res_dir = paste0(fp_res, folder_name)
  dir.create(res_dir, showWarnings = FALSE)

  # Create a subdirectory for the region-horizon combination
  res_subdir <- paste0(
    res_dir,
    "/targetregion",
    toupper(y_region),
    "-tau",
    ifelse(tau < 10, paste0(0, tau), tau)
  )
  dir.create(res_subdir, showWarnings = FALSE)
  
  # Create file path for predictions
  fp_preds <-
    paste0(
      res_subdir,
      "/targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nlayers", ifelse(nlayers < 10, paste0(0, nlayers), nlayers),
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
      "-nens", ifelse(nens < 100, paste0(0, nens), nens),
      "-", message,
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
        n_layers = nlayers,
        tau_emb = 1,
        nh = nh,
        nu = nu,
        n_ensembles = nens,
        cores = ncores,
        seed = seed
      )

    # Compute predictions
    preds <-
      predict_regional_eesn(
        eesn = eesn,
        x_test = as.matrix(x_test),
        t_test = t_test,
        show_var = show_var
      )

    # Save predictions
    write.csv(preds, fp_preds, row.names = FALSE)

  }

}


# EESN PREDICTIONS --------------------------------------------------------

# Predict using a fitted EESN. Returns both in-sample and out-of sample
# predictions (must input test predictors and dates)
predict_regional_eesn <- function(eesn, x_test, t_test, show_var = FALSE) {

  # Get tau
  tau = eesn[[1]]$params_tuning$tau

  # Training data predictions
  preds_train <-
    map(.x = eesn, .f = predict_esn) |>
    map(
      \(preds) data.frame(pred = preds$preds_ins) |>
        tibble::rownames_to_column(var = "date")
    ) |>
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
    map(
      \(preds) data.frame(pred = preds$preds_oos) |>
        tibble::rownames_to_column(var = "date")
    ) |>
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

  # Forecasts from testing data
  preds_test <-
    map(
      .x = eesn,
      .f = predict_esn,
      x_new = as.matrix(x_test),
      t_new = as.character(t_test)
    ) |>
    map(
      \(preds) data.frame(pred = preds$preds_new)
        |> tibble::rownames_to_column(var = "date")
    ) |>
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

  # Join and compute ensemble prediction
  preds <-
    bind_rows(preds_train, preds_oos, preds_test) |>
    na.omit() |> # remove dates where no predictions are given (due to lags and such)
    mutate(ens_member = paste0("pred", ens_member)) |>
    pivot_wider(names_from = "ens_member", values_from = "pred") |>
    rowwise() |>
    mutate(
      pred = mean(c_across(starts_with("pred")))
    ) |>
    ungroup()

  # Determine which form of predictions to return
  if (show_var) {
    preds = preds |> select(year, week, data, pred, everything())
  } else {
    preds = preds |> select(year, week, data, pred)
  }
  
  # Return the predictions
  return(preds)

}


# COMPUTE EESN RMSES ------------------------------------------------------

# Function for computing RMSEs on predictions from the time series cross
# validation procedure
compute_cv_rmses <- function(
    y_obs,
    y_region,
    n_splits,
    tau,
    nlayers,
    m,
    nh,
    nu,
    alpha = 1,
    fp_res
) {

  # Create general file path
  fp_preds_rmses <-
    paste0(
      "targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nlayers", ifelse(nlayers < 10, paste0(0, nlayers), nlayers),
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
      "-nens", ifelse(nens < 100, paste0(0, nens), nens)
    )
  
  # Create file path for RMSEs
  fp_rmses = paste0(fp_res, "cv-rmses/", fp_preds_rmses, ".csv")

  # If file doesn't already exist...
  if (!file.exists(fp_rmses)) {

    # Specify the number of CV splits
    if (nlayers == 1) {
      splits = 1:n_splits
    } else {
      splits = 4:n_splits
    }
    
    # Load and join CV predictions on validation folds
    cv_val_preds <-
      map(
        .x = splits,
        .f = function(split) {
          preds <-
            read.csv(paste0(
              fp_res,
              "cv-preds/",
              "targetregion",
              toupper(y_region), 
              "-tau", ifelse(tau < 10, paste0(0, tau), tau),
              "/",
              fp_preds_rmses,
              "-split",
              split,
              ".csv"
            )) |>
            filter(data == "testing") |>
            # remove last tau since (technically next CV fold)
            filter(row_number() <= n() - tau) |>
            mutate(cv_val_split = split)
        }) |>
      list_rbind()

    # Join predictions with observational data
    preds_obs <-
      left_join(y_obs, cv_val_preds, by = join_by(year, week)) |>
      filter(data == "testing")

    # Compute RMSEs
    rmses <-
      preds_obs |>
      summarise(
        rmse = sqrt(sum((pred - obs)^2) / n()),
        .by = cv_val_split
      )

    # Compute RMSEs with extreme temps
    rmses_extreme <-
      preds_obs |>
      filter(extreme) |>
      summarise(
        rmse_extreme = sqrt(sum((pred - obs)^2) / n()),
        .by = cv_val_split
      )

    # Compute weighted extreme RMSEs
    rmses_weighted <-
      preds_obs |>
      mutate(weights = 1 + alpha * pmax(0, abs(temp_zscore) - 1)) |>
      summarise(
        rmse_weighted = sqrt(sum(weights * (pred - obs)^2) / sum(weights)),
        .by = cv_val_split
      )
    
    # Join RMSEs
    all_rmses <-
      left_join(rmses, rmses_extreme, by = join_by(cv_val_split)) |>
      left_join(rmses_weighted, by = join_by(cv_val_split)) |>
      mutate(nlayers = nlayers)

    # Save RMSEs
    write.csv(all_rmses, fp_rmses, row.names = FALSE)

  }

}

# Function to compute RMSE with clusters
compute_rmses_subset <- function(
    rmse_data,
    y_region,
    tau,
    m,
    nlayers,
    nh,
    nu,
    nens,
    groups,
    alpha = 1,
    cut_off_year,
    folder_preds,
    folder_rmses,
    fp_res
) {

  # Create file path for predictions for this unique set of tuning parameters
  res_dir = paste0(fp_res, folder_rmses)
  dir.create(res_dir, showWarnings = FALSE)
  rmse_subdir <- 
    paste0(
      res_dir, 
      "/targetregion", toupper(y_region), 
      "-tau", ifelse(tau < 10, paste0(0, tau), tau)
    )
  dir.create(rmse_subdir, showWarnings = FALSE)
  preds_subdir <- 
    paste0(
      fp_res, folder_preds, 
      "/targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau)
    )
  
  run_path <- 
    paste0(
      "/targetregion", toupper(y_region),
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nlayers", ifelse(nlayers < 10, paste0(0, nlayers), nlayers),
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
      "-nens", ifelse(nens < 100, paste0(0, nens), nens),
      "-groups", groups,
      ".csv"
    )
  
  fp_rmses = paste0(rmse_subdir, run_path)
  fp_preds = paste0(preds_subdir, run_path)

  if (!file.exists(fp_rmses)) {
    if (file.exists(fp_preds)) {

      # Load predictions
      preds = read.csv(fp_preds)

      # Join predictions with observational data
      preds_obs <-
        left_join(
          preds,
          rmse_data,
          by = join_by(year, week, data)
        ) |>
        filter(year < cut_off_year)

      # Compute RMSEs
      rmses <-
        preds_obs |>
        summarise(rmse = sqrt(sum((pred - obs)^2) / n()), .by = data)

      # Compute RMSEs with extreme temps
      rmses_extrems <-
        preds_obs |>
        filter(extreme) |>
        summarise(rmse_extreme = sqrt(sum((pred - obs)^2) / n()), .by = data)

      # Compute weighted extreme RMSEs
      rmses_weighted <-
        preds_obs |>
        mutate(weights = 1 + alpha * pmax(0, abs(temp_zscore) - 1)) |>
        summarise(
          rmse_weighted = sqrt(sum(weights * (pred - obs)^2) / sum(weights)),
          .by = data
        )
      
      # Join RMSEs
      all_rmses <-
        left_join(rmses, rmses_extrems, by = join_by(data)) |>
        left_join(rmses_weighted, by = join_by(data))

      # Save RMSEs
      write.csv(all_rmses, fp_rmses, row.names = FALSE)

    }
  }

}


# COMPUTE GROUPED FI ------------------------------------------------------

# Function for calculating PFI for grouped inputs.
# Takes in the EESN model, nreps, clusters_RH (the clusters for the region
# and horizon. Does this without parallelizing so that it is easier to track
# progress and debug if needed
# Written by: Jacob Johnson
compute_gfi_series <- function(model, nreps, clusters_RH) {
  
  # Compute observered data RMSE
  obs = model[[1]]$data_train$y_train
  preds = predict_Eesn(model)
  rmse = sqrt(mean((obs - preds)^2))
  
  # Extract model inputs
  x = model[[1]]$data_input$x
  
  # List of clusters
  cluster_list <- unique(clusters_RH$Group)
  
  # Compute feature importance
  fi <- rep(NA, max(cluster_list))
  pb <- txtProgressBar(min=0, max=length(cluster_list), style=3)
  it <- 0
  for(j in cluster_list){
    
    x_perm = x
    rmse_perm = rep(NA, nreps)
    group_vars = clusters_RH$Feature[clusters_RH$Group == j]
    
    for (k in 1:nreps) {
      for(p in group_vars){
        x_perm[,p] = sample(x[,p], size = dim(x)[1], replace = FALSE)
      }
      preds_perm = predict_Eesn(model = model, x_ood = x_perm)
      rmse_perm[k] = sqrt(mean((obs - preds_perm)^2))
    }
    fi[j] <- rmse - mean(rmse_perm)
    
    it <- it + 1
    setTxtProgressBar(pb, it)
  }
  
  # Return feature importance
  return(fi)
  
}

# Function for calculating PFI for grouped inputs for the Final GPFI
# Takes in the EESN model, nreps, clusters_RH (the clusters for the region
# and horizon. Does this without parallelizing so that it is easier to track
# progress and debug if needed
# Written by: Jacob Johnson
compute_gfi_series_final <- function(model, nreps, clusters_RH) {
  
  # Compute observered data RMSE
  obs = model[[1]]$data_train$y_train
  preds = predict_Eesn(model)
  rmse = sqrt(mean((obs - preds)^2))
  
  # Extract model inputs
  x = model[[1]]$data_input$x
  
  # List of clusters
  cluster_list <- unique(clusters_RH$Group)
  
  # Compute feature importance
  fi <- matrix(NA, nrow=length(cluster_list), ncol=nreps)
  pb <- txtProgressBar(min=0, max=length(cluster_list), style=3)
  it <- 0
  for(j in cluster_list){
    x_perm = x
    rmse_perm = rep(NA, nreps)
    group_vars = clusters_RH$var[clusters_RH$Group == j]
    
    for (k in 1:nreps) {
      for(p in group_vars){
        x_perm[,p] = sample(x[,p], size = dim(x)[1], replace = FALSE)
      }
      preds_perm = predict_Eesn(model = model, x_ood = x_perm)
      rmse_perm[k] = sqrt(mean((obs - preds_perm)^2))
    }
    fi[it+1,] <- rmse - rmse_perm ## Changed to be rmse_perm instead of rmse
    
    it <- it + 1
    setTxtProgressBar(pb, it)
  }
  
  # Return feature importance
  res <- cbind(cluster_list, fi)
  colnames(res) <- c('cluster', paste0('fi_rep', 1:nreps))
  return(res)
  
}

# Function for calculating PFI for grouped inputs.
# Takes in the EESN model, nreps, clusters_RH (the clusters for the region and
# horizon), and the number of cores to use
# Written by: Jacob Johnson
compute_gfi <- function(model, nreps, clusters_RH, ncores = NULL) {
  
  # Compute observered data RMSE
  obs = model[[1]]$data_train$y_train
  preds = predict_Eesn(model)
  rmse = sqrt(mean((obs - preds)^2))
  
  # Extract model inputs
  x = model[[1]]$data_input$x
  
  # Specify number of cores (if null)
  if (is.null(ncores)) {
    ncores = detectCores() - 1  
  }
  
  # List of clusters
  cluster_list <- unique(clusters_RH$Group)
  
  # Compute feature importance
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  fiList <- foreach::foreach(j=cluster_list, .packages='listenr', .verbose=T) %dopar% {
    x_perm = x
    rmse_perm = rep(NA, nreps)
    group_vars = clusters_RH$Feature[clusters_RH$Group == j]
    for (k in 1:nreps) {
      for(p in group_vars){
        x_perm[,p] = sample(x[,p], size = dim(x)[1], replace = FALSE)
      }
      preds_perm = predict_Eesn(model = model, x_ood = x_perm)
      rmse_perm[k] = sqrt(mean((obs - preds_perm)^2))
    }
    rmse - mean(rmse_perm)
  }
  
  parallel::stopCluster(cl)
  
  # Return feature importance
  return(unlist(fiList))
  
}

