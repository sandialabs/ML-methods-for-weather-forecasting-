library(dplyr)
library(ggplot2)
library(forcats)
library(tidytext)
library(ggpubr)
library(patchwork)
# esn all data Region Horizon  RMSE RMSE_extreme Num_features
resesn= read.csv("final_results/Chosenby_AllData-final-rmses.csv")
resesn$X = NULL
resesn = resesn %>% filter(data == "testing") %>%
  select(-file, -m, -nh, -nu, -nens, -groups, -data) %>%
  rename(RMSE = rmse,
         RMSE_extreme = rmse_extreme,
         Region = target_region, 
         Horizon = tau) %>%
  select(Region, Horizon, RMSE, RMSE_extreme) %>%
  mutate(model_type = "ESN")
# RF all data
resrf = read.csv("final_results/RF_rmse_results.csv")
resrf$Feature_names = NULL
resrf$Num_features = NULL
resrf$model_type = "RF"

linmod_results = read.csv("final_results/test_rmse_results2021_2024.csv")
names(linmod_results) = c("Region", "RMSE", "RMSE_extreme")
linmod_results = rbind(linmod_results, linmod_results, linmod_results, linmod_results)
linmod_results$Horizon = rep(1:4, each=5)
linmod_results = linmod_results[,c(1,4,2,3)]
linmod_results$model_type = "Linear_model"

persist = read.csv("final_results/rmse_results_persistence2021_2024.csv")
persist$model_type = "Persistence"

data = rbind(resesn, resrf, linmod_results, persist)

p1=ggplot(data = data, aes(x = Horizon, y = RMSE, color = model_type)) +
  geom_line() + 
  geom_point() +
  facet_wrap(~Region, nrow = 1, ncol = 5) + 
  theme_minimal(base_size = 20) +
  ylab("RMSE") + 
  #ggtitle("Best mean models") +
   ylim(c(1.5, 8)) +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        text = element_text(size=20),
        legend.title = element_blank()) +
  scale_color_manual(values = c("RF" = "turquoise", "Persistence" = "violet", "Linear_model" = "darkgray",
                                "ESN" = "darkorange"))

# now do the same for models chosen according to extreme rmse
resesn= read.csv("final_results/Chosenby_Extremes-final-rmses.csv")
resesn$X = NULL
resesn = resesn %>% filter(data == "testing") %>%
  select(-file, -m, -nh, -nu, -nens, -groups, -data) %>%
  rename(RMSE = rmse,
         RMSE_extreme = rmse_extreme,
         Region = target_region, 
         Horizon = tau) %>%
  select(Region, Horizon, RMSE, RMSE_extreme) %>%
  mutate(model_type = "ESN")
# RF all data
resrf = read.csv("final_results/RF_rmse_results_ChooseEXTREME.csv")
resrf$Feature_names = NULL
resrf$Num_features = NULL
resrf$model_type = "RF"

linmod_results = read.csv("final_results/test_rmse_results2021_2024.csv")
names(linmod_results) = c("Region", "RMSE", "RMSE_extreme")
linmod_results = rbind(linmod_results, linmod_results, linmod_results, linmod_results)
linmod_results$Horizon = rep(1:4, each=5)
linmod_results = linmod_results[,c(1,4,2,3)]
linmod_results$model_type = "Linear_model"

persist = read.csv("final_results/rmse_results_persistence2021_2024.csv")
persist$model_type = "Persistence"

data_extreme = rbind(resesn, resrf, linmod_results, persist)

data2 = data
data_extreme2 = data_extreme
data2$RMSE_extreme = NULL
data2$Type = "All"

data_extreme2$RMSE = NULL
colnames(data_extreme2) = c("Region", "Horizon", "RMSE", "model_type")
data_extreme2$Type = "Extremes"
data_all = rbind(data2, data_extreme2)
#write.csv(data_all, "data_all.csv")
ggplot(data = data_all, aes(x = Horizon, y = RMSE, color = model_type)) +
  geom_line() +
  geom_point() +
  facet_grid(Type ~ Region, scales="free_y") +
  theme_minimal(base_size = 16) +
  theme(
    text = element_text(size = 20),
    strip.text = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    panel.spacing = unit(1, "lines"),
    strip.background = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  ylab("RMSE") +
  scale_color_manual(
    values = c(
      "RF" = "turquoise",
      "Persistence" = "violet",
      "Linear_model" = "darkgray",
      "ESN" = "darkorange"
    ),
    labels = c(
      "RF" = "RF",
      "Persistence" = "Persistence",
      "Linear_model" = "Linear model",   # updated legend label
      "ESN" = "ESN"
    )
  )
ggsave("finalRMSEplot.pdf", width = 3000, height=2400, units = "px")

ggplot(data = data_all, aes(x = Num_Groups, y = Test_RMSE, color = factor(Horizon))) +
  geom_line() +
  geom_point(data = combined_dat,
             aes(x = Num_Groups, y = Test_RMSE),
             size = 3, shape = 16) +
  facet_grid(Model ~ Region, scales="free_y") +  # Force same Region alignment across models
  scale_x_continuous(trans = "log2", limits = c(1, 400)) +
  ylab("RMSE") +
  xlab("Num. Groups") +
  theme_minimal(base_size = 16) +
  theme(
    text = element_text(size = 20),
    strip.text = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    panel.spacing = unit(1, "lines"),
    strip.background = element_blank(),
    legend.position = "bottom"
  ) +
  scale_color_discrete(name = "Horizon")
ggsave("iterativeALLDATA.pdf", width = 3000, height=2400, units = "px")

head(data2)
iter_ESN=read.csv("final_results/group-rmses.csv")
iter_ESN = iter_ESN %>% filter(data == "testing") %>%
  select(-file, -m, -nh, -nu, -nens, -data) %>%
  rename(Test_RMSE = rmse,
         Test_RMSE_extreme = rmse_extreme,
         Region = target_region, 
         Horizon = tau,
         Num_Groups = groups) %>%
  select(Region, Horizon, Test_RMSE, Test_RMSE_extreme, Num_Groups) 
head(iter_ESN)

iter_RF = read.csv("Iterative_GPFI/iterative_group_rmse_results_RF_TS.csv") 
iter_RF$Feature_names=NULL
head(iter_RF)

# Function to apply per Region/Horizon group
select_simplest_within_tolerance <- function(df_group) {
  min_rmse <- min(df_group$Test_RMSE, na.rm = TRUE)
  df_group %>%
    filter(Test_RMSE <= min_rmse * (1 + tolerance)) %>%
    arrange(Num_Groups) %>%
    slice(1)
}

select_simplest_within_tolerance_extreme <- function(df_group) {
  min_rmse <- min(df_group$Test_RMSE_extreme, na.rm = TRUE)
  df_group %>%
    filter(Test_RMSE_extreme <= min_rmse * (1 + tolerance)) %>%
    arrange(Num_Groups) %>%
    slice(1)
}


tolerance=0.01
make_data_all = function(results, code = "default"){
  myres <- results %>%
    group_by(Region, Horizon) %>%
    slice_min(Test_RMSE, with_ties = FALSE) %>%
    #select(Region, Num_Features, Num_Groups, Test_RMSE)
    select(Region, Num_Groups, Test_RMSE)
  
  best_models <- results %>%
    group_by(Region, Horizon) %>%
    group_modify(~ select_simplest_within_tolerance(.x)) %>%
    ungroup()
  

  df=best_models
  df$code = code
  df = select(df, "Region", "Horizon", "Test_RMSE", "code", "Num_Groups")
  return(df)
  
}

make_data_extreme = function(results, code = "default"){
  myres <- results %>%
    group_by(Region, Horizon) %>%
    slice_min(Test_RMSE_extreme, with_ties = FALSE) %>%
    #select(Region, Num_Features, Num_Groups, Test_RMSE)
    select(Region, Num_Groups, Test_RMSE)
  
  
  best_models_extreme <- results %>%
    group_by(Region, Horizon) %>%
    group_modify(~ select_simplest_within_tolerance_extreme(.x)) %>%
    ungroup()
  
  df=best_models_extreme
  df$Test_RMSE_extreme = best_models_extreme$Test_RMSE_extreme
  df$code = code
  df = select(df, "Region", "Horizon", "Test_RMSE_extreme", "code", "Num_Groups")
  return(df)
  
}
ESN_dat_all = make_data_all(iter_ESN, code = "ESN")
RF_dat_all = make_data_all(iter_RF, code = "RF")
ESN_dat_ex = make_data_extreme(iter_ESN, code="ESN")
RF_dat_ex = make_data_extreme(iter_RF, code="RF")

iter_ESN$Model = "ESN"
iter_RF$Model = "RF"
iter_RF = iter_RF %>% select(c("Region" ,"Horizon", "Test_RMSE", 
                               "Test_RMSE_extreme" ,"Num_Groups" ,"Model"))
MLiterDat = rbind(iter_ESN, iter_RF)
MLiterDat$Model = factor(MLiterDat$Model, levels = c("ESN", "RF"))

ESN_dat_all$Model = "ESN"
RF_dat_all$Model = "RF"
combined_dat = rbind(ESN_dat_all, RF_dat_all)
ggplot(data = MLiterDat, aes(x = Num_Groups, y = Test_RMSE)) +
  geom_line(aes(color = factor(Horizon))) +
  geom_point(
    data = combined_dat,
    aes(x = Num_Groups, y = Test_RMSE, fill = factor(Horizon)),
    size = 3,
    shape = 21,          # filled circle with border
    color = "black",     # outline color
    stroke = 0.3         # outline thickness (thin)
  ) +
  facet_grid(Model ~ Region, scales = "free_y") +
  scale_x_continuous(trans = "log2", limits = c(1, 400)) +
  ylab("RMSE") +
  xlab("Num. Groups") +
  theme_minimal(base_size = 16) +
  theme(
    text = element_text(size = 20),
    strip.text = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    panel.spacing = unit(1, "lines"),
    strip.background = element_blank(),
    legend.position = "bottom"
  ) +
  scale_color_discrete(name = "Horizon") +
  scale_fill_discrete(name = "Horizon")
ggsave("iterativeALLDATA.pdf", width = 3000, height=2400, units = "px")

str(combined_dat_ex)
str(combined_dat)
ESN_dat_ex$Model = "ESN"
RF_dat_ex$Model = "RF"
str(MLiterDat)
combined_dat_ex= rbind(ESN_dat_ex, RF_dat_ex)
ggplot(data = MLiterDat, aes(x = Num_Groups, y = Test_RMSE_extreme, color = factor(Horizon))) +
  geom_line() +
  geom_point(data = combined_dat_ex,
             aes(x = Num_Groups, y = Test_RMSE_extreme, fill = factor(Horizon)),
             size = 3, 
             shape = 21, 
             color = "black",
             stroke = 0.3) +
  facet_grid(Model ~ Region, scales="free_y") +  # Force same Region alignment across models
  scale_x_continuous(trans = "log2") +
  #ylim(1.5, 3.3) +
  ylab("RMSE") +
  xlab("Num. Groups") +
  theme_minimal(base_size = 16) +
  theme(
    text = element_text(size = 20),
    strip.text = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    panel.spacing = unit(1, "lines"),
    strip.background = element_blank(),
    legend.position = "bottom"
  ) +
  scale_color_discrete(name = "Horizon") +
  scale_fill_discrete(name = "Horizon")
ggsave("iterativeEXTREMES.pdf", width = 3000, height=2400, units = "px")

