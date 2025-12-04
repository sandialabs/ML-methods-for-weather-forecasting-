########################################
########################################
## 03.1-FinalTestSet-W.R              ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript to fit EESN over W         ##
## - Uses 1980-2020 as training data  ##
##   with the goal of predicting the  ##
##   final holdout years 2021-2024    ##
########################################
########################################

# SETUP -------------------------------------------------------------------
## Load set up
source("../03.0-Setup/03-setup.R")

##Load functions
source("../03.0-Setup/03-functions.R")

## Load Initial FI Results
fi_initial <- fi_initial <- readr::read_csv(
  paste0(fp,
         "agu-manuscript-code/EESNs/02-grouped-feature-importance/results/gpfi-initial.csv"))


# Best Models Based on Data -----------------------------------------------
## Specify target regions
target_regions = "W"
input_vars <- "Chosenby_AllData"
for (this_target_region in target_regions) {
  
  for(this_tau in taus){
    
    ## Status Message
    print(paste0("Target Region:", this_target_region, " Tau:", this_tau))
    
    ## FI for this region and horizon
    fi_RH <- fi_initial |> 
      filter(tau == this_tau,
             region == this_target_region) |> 
      select(-region, -tau) 
    
    ## Specify m
    m <- fi_RH$m[1]
    
    ## Specify nu
    nu <- fi_RH$nu[1] / 100
    
    ## Specify nh
    nh <- fi_RH$nh[1]
    
    ## Load the Best models
    best_models_all <- read.csv('../../02-grouped-feature-importance/results/best-models-all.csv')
    
    ## Find the number of groups used in the best model
    g <- (best_models_all |> 
            filter(tau == this_tau,
                   target_region == this_target_region,
                   data == 'training'))$best_model_all
    
    
    ## List of variables from a subset of groups
    remove_vars <- (fi_RH |>
                      filter(rank > g))$var
    
    ## Extract target variables
    targets_train <-  outputs_train |> select(all_of(this_target_region))
    colnames(targets_train) <- paste0(this_target_region, "_current_week")
    targets_test <- outputs_test |> select(all_of(this_target_region))
    colnames(targets_test) <- paste0(this_target_region, "_current_week")
    
    ## Separate inputs into train/test and convert to matrices
    inputs_train <-
      climate_inputs |> 
      filter(year < test_year) |>
      select(-year, -week, -date) |>
      bind_cols(targets_train)
    colnames(inputs_train) <- str_remove_all(colnames(inputs_train), "_weekave")
    
    inputs_test <-
      climate_inputs |> 
      filter(year >= test_year) |>
      select(-year, -week, -date) |>
      bind_cols(targets_test)
    colnames(inputs_test) <- str_remove_all(colnames(inputs_test), "_weekave")
    
    ## Only include variables in the subset
    inputs_train_subset <- inputs_train |>
      select(-all_of(remove_vars))
    inputs_test_subset <- inputs_test |>
      select(-all_of(remove_vars))
    
    ## Fit model and save predictions
    fit_regional_inputs_eesns_subset(
      y_train = targets_train,
      y_test = targets_test,
      y_region = this_target_region,
      x_train = inputs_train_subset,
      x_test = inputs_test_subset,
      x_regions = input_regions,
      x_vars = input_vars,
      t_train = dates_train,
      t_test = dates_test,
      tau = this_tau,
      m = m,
      nh = nh,
      nu = nu,
      nens = nens,
      seed = seed,
      ncores = ncores, 
      groups = g
    )
    
  }
  
}



# Best Models based on Extremes -------------------------------------------
input_vars <- "Chosenby_Extremes"
for (this_target_region in target_regions) {
  
  for(this_tau in taus){
    
    ## Status Message
    print(paste0("Target Region:", this_target_region, " Tau:", this_tau))
    
    ## FI for this region and horizon
    fi_RH <- fi_initial |> 
      filter(tau == this_tau,
             region == this_target_region) |> 
      select(-region, -tau) 
    
    ## Specify m
    m <- fi_RH$m[1]
    
    ## Specify nu
    nu <- fi_RH$nu[1] / 100
    
    ## Specify nh
    nh <- fi_RH$nh[1]
    
    ## Load the Best models
    best_models_extreme <- read.csv('../../02-grouped-feature-importance/results/best-models-extreme.csv')
    
    ## Find the number of groups used in the best model
    g <- (best_models_extreme |> 
            filter(tau == this_tau,
                   target_region == this_target_region,
                   data == 'training'))$best_model_extreme
    
    
    ## List of variables from a subset of groups
    remove_vars <- (fi_RH |>
                      filter(rank > g))$var
    
    ## Extract target variables
    targets_train <-  outputs_train |> select(all_of(this_target_region))
    colnames(targets_train) <- paste0(this_target_region, "_current_week")
    targets_test <- outputs_test |> select(all_of(this_target_region))
    colnames(targets_test) <- paste0(this_target_region, "_current_week")
    
    ## Separate inputs into train/test and convert to matrices
    inputs_train <-
      climate_inputs |> 
      filter(year < test_year) |>
      select(-year, -week, -date) |>
      bind_cols(targets_train)
    colnames(inputs_train) <- str_remove_all(colnames(inputs_train), "_weekave")
    
    inputs_test <-
      climate_inputs |> 
      filter(year >= test_year) |>
      select(-year, -week, -date) |>
      bind_cols(targets_test)
    colnames(inputs_test) <- str_remove_all(colnames(inputs_test), "_weekave")
    
    ## Only include variables in the subset
    inputs_train_subset <- inputs_train |>
      select(-all_of(remove_vars))
    inputs_test_subset <- inputs_test |>
      select(-all_of(remove_vars))
    
    ## Fit model and save predictions
    fit_regional_inputs_eesns_subset(
      y_train = targets_train,
      y_test = targets_test,
      y_region = this_target_region,
      x_train = inputs_train_subset,
      x_test = inputs_test_subset,
      x_regions = input_regions,
      x_vars = input_vars,
      t_train = dates_train,
      t_test = dates_test,
      tau = this_tau,
      m = m,
      nh = nh,
      nu = nu,
      nens = nens,
      seed = seed,
      ncores = ncores, 
      groups = g
    )
    
  }
  
}



