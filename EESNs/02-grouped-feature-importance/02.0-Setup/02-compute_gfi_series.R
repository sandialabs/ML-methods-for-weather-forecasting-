########################################
########################################
## 02-compute_gfi_series.R            ##
## Jacob Johnson                      ##
## ---------------------------------- ##
## Rscript containing a function to   ##
## calculate PFI for grouped inputs.  ##
## - Takes in the EESN model, nreps,  ##
##   clusters_RH (the clusters for    ##
##   the region and horizon)          ##
## - Does this without parallelizing  ##
##   so that it is easier to track    ##
##   progress and debug if needed     ##
########################################
########################################

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
