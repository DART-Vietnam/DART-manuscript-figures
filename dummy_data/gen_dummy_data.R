library(tidyverse)
library(worcs)

raw_full_inc <- read_csv("cleaned_incidence_data.csv")

set.seed(764)

dummy_dat <- raw_full_inc %>%
  as_tibble() %>%
  na.omit() %>%
  slice_sample(n = 2, by = c(date, district)) %>%
  mutate(
    across(where(is.character), factor),
    age = as.integer(age),
    date = as.numeric(date)
  ) %>%
  synthetic(
    model_expression = NULL,
    predict_expression = sample(y, size = length(y), replace = TRUE)
  ) %>%
  mutate(date = as.Date(date))
summary(dummy_dat)

dummy_dat %>% write_csv("dummy_data/dummy_epi_data.csv")
