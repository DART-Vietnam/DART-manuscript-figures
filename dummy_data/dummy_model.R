library(tidyverse)
library(stars)
library(worcs)
library(fable)
library(tsibble)
library(patchwork)

# Data ingestion
## Read GID lookup dataframe
gid_lookup_df <- read_rds("dummy_data/gid2_lookup_df.rds")

## Read ERA5 weather data from DART-Pipeline
weekly_weather_df <- read_ncdf(
  "dummy_data/VNM-2-2001-2019-era5.nc",
  make_units = FALSE
) %>%
  as_tibble() %>%
  mutate(region = as.character(region), time = as.Date(time)) %>%
  filter(startsWith(as.character(region), "VNM.25"))

## Read ECMWF weather forecast data from DART-Pipeline
weather_fcst_df <- read_ncdf(
  "dummy_data/VNM-2-2025-07-11-ecmwf.forecast.nc",
  make_units = FALSE
) %>%
  as_tibble() %>%
  mutate(region = as.character(region), date = as.Date(date)) %>%
  filter(startsWith(as.character(region), "VNM.25"))

## Read dummy incidence data
dummy_dat <- read_csv("dummy_data/dummy_epi_data.csv") %>%
  filter(
    hospital %in% c("HTD", "CH1", "CH2"),
    in_out_patient %in% c("in-patient", "discharged")
  )

# Preprocess data
## Truncate incidence by weather temporal coverage
min_date <- min(weekly_weather_df$time)
max_date <- max(weekly_weather_df$time)

## Aggregate line-listing incidence to daily incidence
spatagged_daily_incidence <- dummy_dat %>%
  select(date, district) %>%
  # time series length restricting
  filter(between(date, min_date, max_date)) %>%
  # spatial unit aggregation
  group_by(district, date) %>%
  tally() %>%
  # 0-filling missing dates
  complete(
    date = seq.Date(min_date, max_date, by = "1 day"),
    fill = list(n = 0)
  ) %>%
  ungroup() %>%
  # switch to using GID instead of semantic names
  left_join(
    gid_lookup_df,
    by = join_by(district == VARNAME_2)
  ) %>%
  select(-c(district)) %>%
  relocate(GID_2, .before = date)

## Bind incidence and weather data
daily_inc_weather <- spatagged_daily_incidence %>%
  left_join(
    weekly_weather_df,
    by = join_by(GID_2 == "region", date == time)
  ) %>%
  mutate(
    isoyear = as.integer(isoyear(date)),
    isoweek = as.integer(isoweek(date)),
    .after = date
  ) %>%
  rename(region = GID_2)

weekly_inc_weather <- daily_inc_weather %>%
  group_by(region, isoweek, isoyear) %>%
  summarise(
    # grouping variable
    date = min(date),
    # incidence
    n = sum(n),
    # variables that work better with mean
    across(
      starts_with(
        c(
          "t2m",
          "r",
          "q",
          "mn2t24",
          "mx2t24",
          "mnr24",
          "mxr24",
          "mnq24",
          "mxq24",
          "spi",
          "spei"
        )
      ),
      ~ mean(.x, na.rm = TRUE)
    ),
    # variables that work better with sum
    across(starts_with(c("tp", "hb")), ~ sum(.x, na.rm = TRUE))
  ) %>%
  ungroup()

train_ts_df <- weekly_inc_weather %>%
  filter(date < as.Date("2019-01-01")) %>%
  mutate(date = yearweek(date)) %>%
  as_tsibble(key = region, index = date)

test_ts_df <- weekly_inc_weather %>%
  filter(date >= as.Date("2019-01-01")) %>%
  mutate(date = yearweek(date)) %>%
  as_tsibble(key = region, index = date)

sarimax_results <- train_ts_df %>%
  model(sarimax = ARIMA(log1p(n) ~ t2m + tp + r))

sarimax_results %>% filter(region == "VNM.25.10_1") %>% report()

forecast_results <- sarimax_results %>%
  forecast(test_ts_df)

p1 <- forecast_results %>%
  autoplot(test_ts_df) +
  facet_wrap(~region, ncol = 4, scales = "free_y")

r2_df <- sarimax_results %>%
  fitted() %>%
  na.omit() %>%
  as_tibble() %>%
  left_join(
    train_ts_df %>% slice_head(n = -4, by = region) %>% select(region, date, n)
  )

p2 <- r2_df %>%
  ggplot(aes(x = n, y = .fitted)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~region, ncol = 4, scales = "free")
combined_plot <- (p1 / p2) + plot_annotation(tag_levels = "a")

ggsave(
  filename = "FigS3.tiff",
  plot = combined_plot,
  width = 15,
  height = 24,
  units = "in",
  dpi = 300,
  compression = "lzw"  # recommended for TIFFs
)
