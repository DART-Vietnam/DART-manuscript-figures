# import necessary packages
library(geodata)
library(tidyverse)
library(sf)

# PLEASE UNCOMMENT BELOW AND SET THE WORKING DIRECTORY TO **THIS FOLDER**
# setwd("/absolute/path/to/dart-manuscript-figures")

# set ggplot2 theme
theme_set(theme_bw())

# download shapefile data from GADM to `./gadm`
gadm(country = "VNM", level = 2, path = ".", version = "4.1")

# read in shapefile
vnm_shp <- read_rds("gadm/gadm41_VNM_2_pk.rds") %>%
  terra::unwrap() %>%
  st_as_sf()

# filter for HCMC
hcmc_shp <- vnm_shp %>%
  filter(startsWith(GID_2, "VNM.25"))

# plot
p <- hcmc_shp %>%
  ggplot() +
  geom_sf() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )


# export figure
ggsave("FigS1.tiff", plot = p, scale = 2)
