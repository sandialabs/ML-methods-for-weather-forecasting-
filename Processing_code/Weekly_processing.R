library(dplyr)
library(lubridate)
library(purrr)
library(stringr)
library(tidyr)
library(data.table)

setwd("/Volumes/Projects/subseasonal_extreme/weekly_ave_test")
Conus_regions = read.csv("CONUS_Regions_1980_to_2020.csv")
time_inputs = read.csv("time_inputs_1980_to_2020.csv")
head(time_inputs)

fp = "/Volumes/Projects/subseasonal_extreme/weekly_ave_test/regional_inputs/"
list.files(fp)
# Get target data files for all CONUS
regional_input_files = list.files(fp)[str_detect(list.files(fp), "PC20")]
stringr::str_remove(regional_input_files[1], "_PC20_1980_to_2020.csv")
# Determine names of Northern Hemisphere inputs
input_vars = tolower(stringr::str_remove(regional_input_files,"_PC20_1980_to_2020.csv"))
input_vars
# Load and clean Northern Hemisphere inputs
reg_inputs <-
  set_names(regional_input_files) |>
  map(
    .f = function(x)
      read.csv(paste0(fp, x))
  )  |>
  list_rbind(names_to = "file") |>
  mutate(
    file = str_remove(file, "_PC20_1980_to_2020.csv"),
  ) |>
  pivot_wider(
    id_cols = c("Date"),
    names_from = "file",
    values_from = Mean:PC20, 
    names_glue = "{file}_{.value}"
  ) |>
  rename_all(tolower)
  #mutate(date = ymd(date)) |>
  #select(date, mean, contains(input_vars))

fwrite(reg_inputs, "weekly_aves_regional_inputs_1980_2020.csv")
head(reg_inputs)
colnames(reg_inputs)
test = read.csv("weekly_aves_regional_inputs_1980_2020.csv")
test$Date = test$date
head(test[,1:5])


# Processing for the test data, which will be appended to 1980-2020 data

fp = "/Volumes/Projects/subseasonal_extreme/weekly_ave_test/test_inputs/"
list.files(fp)
# Get target data files for all CONUS
regional_input_files = list.files(fp)[str_detect(list.files(fp), "PC20")]
stringr::str_remove(regional_input_files[1], "_PC20_2021_to_2024.csv")
# Determine names of Northern Hemisphere inputs
input_vars = tolower(stringr::str_remove(regional_input_files,"_PC20_2021_to_2024.csv"))
input_vars
# Load and clean Northern Hemisphere inputs
reg_inputs <-
  set_names(regional_input_files) |>
  map(
    .f = function(x)
      read.csv(paste0(fp, x))
  )  |>
  list_rbind(names_to = "file") |>
  mutate(
    file = str_remove(file, "_PC20_2021_to_2024.csv"),
  ) |>
  pivot_wider(
    id_cols = c("Date"),
    names_from = "file",
    values_from = Mean:PC20, 
    names_glue = "{file}_{.value}"
  ) |>
  rename_all(tolower)
#mutate(date = ymd(date)) |>
#select(date, mean, contains(input_vars))

fwrite(reg_inputs, "weekly_aves_regional_inputs_2021_2024.csv")
test = read.csv("weekly_aves_regional_inputs_2021_2024.csv")
head(test)

week_aves19802020 = read.csv("weekly_aves_regional_inputs_1980_2020.csv")
head(week_aves19802020)
week_aves_1980_2024 = rbind(week_aves19802020, reg_inputs)
fwrite(week_aves_1980_2024, "weekly_aves_regional_inputs_1980_2024.csv")
