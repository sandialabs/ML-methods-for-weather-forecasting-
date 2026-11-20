## Rscript to calculate PFI for the 
## groups for clusters of related
## inputs on the final models
## using a small number of inputs


# Script Options ----------------------------------------------------------

this_target_region = 'SW'
taus = 1:4


# SETUP -------------------------------------------------------------------

# Specify data type
data = "test"

# Load set up
source("../setup.R")
source("../functions.R")

# Load initial FI results
fi_initial = readr::read_csv('../results/gpfi-initial.csv')

# Load best number of groups
best_models_all = read.csv('../results/best-subset-models-all.csv')


# DATA STEPS --------------------------------------------------------------

for(this_tau in taus){
  
  # Status Message
  print(paste0("Target Region:", this_target_region, " Tau:", this_tau))
  
  # FI for this region and horizon
  fi_RH <- 
    fi_initial |> 
    filter(
      tau == this_tau,
      region == this_target_region
    ) |> 
    select(-region, -tau) 
  
  # Specify m
  m = fi_RH$m[1]
  
  # Specify nu
  nu = fi_RH$nu[1] / 100
  
  # Specify number of layers
  nlayers = fi_RH$nlayers[1]
  
  # Specify nh
  nh = fi_RH$nh[1]
  
  # Find the number of groups used in the best model
  g <- 
    (best_models_all |> 
       filter(
         tau == this_tau, 
         target_region == this_target_region,
         data == 'training')
    )$best_model_all
  
  # List of variables from a subset of groups
  remove_vars = (fi_RH |> filter(rank > g))$var
  groups_to_keep_RH = fi_RH |> filter(rank <= g)
  
  # Extract target variables
  targets_train_val = outputs_train_val |> select(all_of(this_target_region))
  colnames(targets_train_val) = paste0(this_target_region, "_current_week")
  
  # Separate inputs into train/test and convert to matrices
  inputs_train_val <-
    inputs_train_val_test |> 
    filter(year < test_year) |>
    select(-year, -week, -date) |>
    bind_cols(targets_train_val)
  colnames(inputs_train_val) = str_remove_all(colnames(inputs_train_val), "_weekave")
  
  # Only include variables in the subset
  inputs_train_val_subset <-
    inputs_train_val |>
    select(-all_of(remove_vars))
  
  # Extract dates
  dates_train_val = sort(unique(outputs_train_val$date))
  
  # FEATURE IMPORTANCE ------------------------------------------------------
  
  # Number of reps for feature importance
  nreps = 30
  
  # Prepare Directory Path
  fi_dir <-
    paste0(
      "../results/final-gpfi",
      "-nreps", ifelse(nreps < 10, paste0(0, nreps), nreps), "/"
    )
  dir.create(fi_dir)
  
  # Specify current region and tau
  target_region = this_target_region
  tau = this_tau
  
  # Prepare file path
  fi_file <- 
    paste0(
      fi_dir,
      "targetregion", target_region, 
      "-tau", ifelse(tau < 10, paste0(0, tau), tau),
      "-m", ifelse(m < 10, paste0(0, m), m),
      "-nlayers", ifelse(nlayers < 10, paste0(0, nlayers), nlayers),
      "-nh", ifelse(nh < 1000, paste0(0, nh), nh),
      "-nu", ifelse(nu == 0, "000", ifelse(nu == 1, nu*100, str_remove(paste0(0, nu*100), "\\."))),
      "-groups", g, 
      ".csv"
    )
  
  if (!file.exists(fi_file)) {
    
    # Fit EESN
    eesn <-
      fit_Eesn(
        x = inputs_train_val_subset |> as.matrix(),
        y = targets_train_val |> as.matrix(),
        t = as.character(dates_train_val),
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
    
    # Compute FI
    fi <-
      compute_gfi_series_final(
        model = eesn,
        nreps = nreps,
        clusters_RH = groups_to_keep_RH
      )
    
    # Save FI
    write.csv(
      x = data.frame(fi),
      file = fi_file,
      row.names = FALSE
    ) 
    
  }
  
}
