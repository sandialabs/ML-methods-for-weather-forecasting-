########################################
########################################
## 02-compute_gfi.R                   ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript containing a function to   ##
## calculate PFI for grouped inputs.  ##
## - Takes in the EESN model, nreps,  ##
##   clusters_RH (the clusters for    ##
##   the region and horizon), and the ##
##   number of cores to use           ##
########################################
########################################

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
