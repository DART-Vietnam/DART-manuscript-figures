# import necessary packages
library(tidyverse)

# PLEASE UNCOMMENT BELOW AND SET THE WORKING DIRECTORY TO **THIS FOLDER**
# setwd("/absolute/path/to/dart-manuscript-figures")

# ingest data
cleaned_incidence_dat <- read_csv("cleaned_incidence_data.csv")

# set min and max date for data filling
min_date <- min(cleaned_incidence_dat$date)
max_date <- max(cleaned_incidence_dat$date)

# do a bit of data transformation
raw_ts_plot_dat <- cleaned_incidence_dat %>%
  group_by(district, date) %>%
  tally() %>%
  # fill out the time series, i.e. days without incidence will have 0
  complete(
    date = seq.Date(min_date, max_date, by = "1 day"),
    fill = list(n = 0)
  ) %>%
  # get year and ISO week of date
  mutate(
    year = year(date),
    isoweek = isoweek(date),
  ) %>%
  # group by and summarise incidence into each year + ISO week
  group_by(year, isoweek) %>%
  summarise(date = min(date), n = sum(n))

# plot
p <- raw_ts_plot_dat %>%
  ggplot(aes(x = date, y = n)) +
  # red background for COVID 19 lockdown period
  annotate(
    "rect",
    xmin = as.Date("2020/01/01"),
    xmax = as.Date("2022/01/01"),
    ymin = 0,
    ymax = max(raw_ts_plot_dat$n),
    fill = "red",
    alpha = 0.5
  ) +
  annotate(
    "label",
    x = as.Date("2018/08/15"),
    y = 3400,
    label = "COVID-19\nlockdowns",
    color = "#ff4946"
  ) +
  # time series as a stairstep plot
  geom_step() +
  # blue dashed y-line as for Circular 54 reporting system
  annotate(
    "segment",
    x = as.Date("2017/01/01"),
    y = 0,
    yend = max(raw_ts_plot_dat$n),
    linetype = 2,
    color = "blue"
  ) +
  annotate(
    "label",
    x = as.Date("2015/01/01"),
    y = 3400,
    label = "Circular\n54/2015/TT-BYT",
    color = "blue"
  ) +
  # axes settings
  scale_y_continuous("Weekly reported dengue incidence") +
  scale_x_date(
    "Date",
    breaks = seq.Date(
      as.Date("2000/01/01"),
      as.Date("2023/01/01"),
      by = "1 year"
    ),
    # date_breaks = "1 year",
    date_labels = "%Y"
  )

# export figure
ggsave("FigS2.tiff", plot = p, width = 10)
