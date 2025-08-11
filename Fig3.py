# Importing modules

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import xarray as xr


path_e = ""  # Path where the observational datasets are included

# In this script we will measure the anomaly correlation coefficient.

# First we load corrected and uncorrected forecast data

data_corr_s = xr.open_dataset("data_corr_s_github.nc", decode_timedelta=True)
data_uncorr_s = xr.open_dataset("data_uncorr_s_github.nc", decode_timedelta=True)

# load of weekly observational data

era_week1 = xr.open_dataset("era_week1_south_github.nc")
era_week2 = xr.open_dataset("era_week2_south_github.nc")

# We set up the variables to process
variables_to_process = ["t2m", "r", "tp"]

# First we need a xarray with which we can measure the climatology


def calculate_climatology_multi_var(
    initial_year, final_year, ensemble_data, observations, variables=["t2m", "r", "tp"]
):
    """Calculate climatology for multiple variables and all ensemble time steps.
    Inputs:

    Initial and final year are  (INT) are used to slice the ERA5 data,

    ensemble_data: forecast data (xarray.Dataset) with valid_time and step dimensions,

    era_ranks_path: Observational data (Xarray.Dataset)

    Variables: (list) with the  variables to process, the on

    Outputs:

    climatology_ds: xarray.Dataset with climatology data for each variable created with the observational data, time is the current
    date of the forecast and number represents the different values of the observation during different years,
    (for example, if the date is 01-01-2003, number==0 represents the actual observation whereas
    number=1 would represent the observation of 01-01-2004, and so on)
    """

    # Load and prepare ERA5 data
    era_ranks = observations
    era_ranks = era_ranks.sel(
        time=slice("%s-01-01" % initial_year, "%s-12-31" % final_year)
    )
    era_ranks = era_ranks.rename({"latitude": "lat", "longitude": "lon"})

    # Apply variable-specific preprocessing
    var_data = {}
    for var in variables:
        var_data[var] = era_ranks[var]  # Default: no conversion

    # Get all unique dates from ensemble data
    all_dates = ensemble_data.valid_time.to_numpy().flatten()
    climatology_datasets = {var: [] for var in variables}

    for date in all_dates:
        date_dt = pd.to_datetime(date)
        day, month, year = date_dt.day, date_dt.month, date_dt.year

        # Find matching dates in other years (excluding the forecast year)
        matching_positions = np.where(
            (era_ranks.time.dt.day == day)
            & (era_ranks.time.dt.month == month)
            & (
                era_ranks.time.dt.year > 2003
            )  # This is because in our sample data, re-forecast data does not go below 2003
            & (era_ranks.time.dt.year != year)
        )[0]

        if len(matching_positions) > 5:
            # Process each variable
            for var in variables:
                weekly_means = []
                for pos in matching_positions:
                    if pos >= 7:  # Ensure we have 7 days before doing the weekly mean
                        if var != "tp":
                            weekly_mean = var_data[var][pos - 7 : pos].mean(dim="time")
                        else:
                            weekly_mean = var_data[var][pos - 7 : pos].sum(dim="time")
                        weekly_means.append(weekly_mean)

                if weekly_means:
                    # Stack all weekly means for this variable
                    clim_data = xr.concat(weekly_means, dim="number")
                    clim_data = clim_data.assign_coords(
                        number=np.arange(len(weekly_means))
                    )

                    # Add time dimension
                    clim_data = clim_data.expand_dims(dim="time", axis=0)
                    clim_data = clim_data.assign_coords(time=[date])

                    climatology_datasets[var].append(clim_data)

    # Concatenate and create final dataset
    final_vars = {}
    for var in variables:
        if climatology_datasets[var]:
            var_climatology = xr.concat(climatology_datasets[var], dim="time")
            final_vars[var] = xr.DataArray(
                data=var_climatology.values,
                dims=["time", "number", "lat", "lon"],
                coords={
                    "time": var_climatology.time,
                    "number": var_climatology.number,
                    "lat": var_climatology.lat,
                    "lon": var_climatology.lon,
                },
            )

    if final_vars:
        climatology_ds = xr.Dataset(final_vars)
        return climatology_ds
    else:
        return None


observations = xr.open_dataset(path_e + "/T2m_r_tp_Vietnam_ERA5v2.nc") # This file is available at zenodo https://zenodo.org/badge/DOI/10.5281/zenodo.15487563.svg

observations["t2m"] = observations["t2m"] - 273.15  # Convert to Celsius
observations["tp"] = observations["tp"] * 1000  # Convert to mm

# Calculate climatology for all variables
variables_to_process = ["t2m", "r", "tp"]

ensemble_data = data_corr_s.sel(step="7.days")
climatology1 = calculate_climatology_multi_var(
    2004, 2020, ensemble_data, observations, variables_to_process
).sel(lon=slice(104.25, 107.5), lat=slice(12, 8.5))

# Climatology of the second forecasting week

ensemble_data = data_corr_s.sel(step="14.days")
climatology2 = calculate_climatology_multi_var(
    2004, 2020, ensemble_data, observations, variables_to_process
).sel(lon=slice(104.25, 107.5), lat=slice(12, 8.5))

# Join climatological datasets inbto 1
climatology_data = [climatology1, climatology2]

# Now we can measure the ACC maps


def calculate_acc_maps(
    data_corrected,
    data_uncorrected,
    era_week1,
    era_week2,
    climatology_datasets,
    p_value=0.05,
):
    """
    Calculate ACC maps for both corrected and uncorrected data.

    Parameters:
    -----------
    data_corrected : xarray.Dataset
        Corrected forecast data
    data_uncorrected : xarray.Dataset
        Uncorrected forecast data
    era_week1 : xarray.Dataset
        ERA5 week 1 observations
    era_week2 : xarray.Dataset
        ERA5 week 2 observations
    climatology_datasets : dict
        Dictionary containing climatology datasets for each variable and week
        Format: {var_name: {week: xarray.Dataset}}
        Example: {'t2m': {1: clim_week1_dataset, 2: clim_week2_dataset}}
    """

    # Configuration
    variables = {
        "t2m": {"levels": [0, 4], "obs_vars": ["q", "r", "tp"], "temp_convert": True},
        "r": {"levels": [4, 8], "obs_vars": ["q", "t2m", "tp"], "temp_convert": False},
        "tp": {"levels": [8, 12], "obs_vars": ["r", "t2m", "q"], "temp_convert": False},
    }

    era_weeks = [era_week1, era_week2]
    lead_labels = ["1 week lead time", "2 week lead time"]
    fontsize = 15
    acc_levels = np.arange(0.6, 1.01, 0.1)

    # Create figure
    fig, axs = plt.subplots(
        3,
        4,
        sharey=True,
        figsize=(13, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axs = axs.flatten()

    def setup_map(ax):
        """Setup map features and styling."""
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(
            cfeature.LAKES, edgecolor="darkblue", facecolor="darkblue", alpha=0.5
        )
        ax.add_feature(cfeature.RIVERS, edgecolor="darkblue", alpha=0.5)
        ax.set_xlim(104.25, 107.5)
        ax.set_ylim(8.5, 12)
        ax.plot(
            106.7,
            10.77,
            "ko",
            markerfacecolor="yellow",
            markersize=7,
            transform=ccrs.Geodetic(),
        )
        ax.text(
            106, 11, "HCMC", transform=ccrs.Geodetic(), fontsize=fontsize - 2, color="k"
        )

    def calculate_acc(forecast_data, obs_data, climatology, var_name):
        """Calculate Anomaly Correlation Coefficient."""
        # Filter for rainy season (as ECMWF forecast data has two dimension, time and valid_time we have to filter both, and separately)
        forecast_filtered = forecast_data.where(
            (forecast_data.time.dt.month > 4) & (forecast_data.time.dt.month < 11),
            drop=True,
        )
        forecast_filtered = forecast_filtered.where(
            (forecast_filtered.valid_time.dt.month > 4)
            & (forecast_filtered.valid_time.dt.month < 11),
            drop=True,
        )
        climatology_filtered = climatology.where(
            (climatology.time.dt.month > 4) & (climatology.time.dt.month < 11),
            drop=True,
        )

        climatology_filtered = climatology_filtered.groupby("time.dayofyear").mean(
            dim="time"
        )

        # Group by day of year
        f = forecast_filtered.squeeze().groupby("valid_time.dayofyear")

        # Filter observations
        obs_filtered = obs_data.where(
            (obs_data.time.dt.month > 4) & (obs_data.time.dt.month < 11), drop=True
        )
        dates_inter = forecast_filtered["valid_time"]
        obs_filtered = obs_filtered.where(
            obs_filtered.time.isin(dates_inter), drop=True
        )
        o = obs_filtered.squeeze().groupby("time.dayofyear")

        # Calculate anomalies
        F, A = f - climatology_filtered, o - climatology_filtered
        F, A = (
            F.where(np.isnan(F) == False, drop=True),
            A.where(np.isnan(A) == False, drop=True),
        )

        # Calculate weights and anomaly correlation coefficient
        weights = np.cos(forecast_data.lat * np.pi / 180).to_numpy()
        weights = weights[np.newaxis, :, np.newaxis]
        axis0 = len(F.time)
        # Repeat weights to match data dimensions
        weights = np.repeat(
            np.repeat(weights, axis=2, repeats=F.lon.size), axis=0, repeats=axis0
        )
        weights = np.reshape(weights, (axis0, F.lat.size, F.lon.size))

        F = F.transpose("time", "lat", "lon")

        var_data_f, var_data_a = F, A[var_name]

        weighted_f = var_data_f - ((weights * var_data_f).sum(dim="time")) / np.sum(
            weights, axis=0
        )
        weighted_a = var_data_a - ((weights * var_data_a).sum(dim="time")) / np.sum(
            weights, axis=0
        )

        numerator = np.sum(
            weights * np.array(weighted_f) * np.array(weighted_a), axis=0
        )
        denom_f = np.sum(
            np.array(weights * (var_data_f - var_data_f.mean(dim="time")) ** 2), axis=0
        )
        denom_a = np.sum(
            np.array(weights * (var_data_a - var_data_a.mean(dim="time")) ** 2), axis=0
        )

        acc = numerator / np.sqrt(denom_f * denom_a)

        # Calculate p-values in which the correlation is significant
        p_vals = np.zeros((len(F.lat), len(F.lon)))
        for la in range(len(F.lat)):
            for lo in range(len(F.lon)):
                if var_name == "tp":
                    p_vals[la, lo] = pearsonr(F[:, la, lo], A.tp[:, la, lo]).pvalue
                else:
                    p_vals[la, lo] = pearsonr(
                        var_data_f[:, la, lo], var_data_a[:, la, lo]
                    ).pvalue

        p_vals = np.where(
            p_vals < p_value, 1, 0
        )  # Here, we replace those p values that below p_value to 0, (p values below p_value shows
        # that observations and forecast are significantly different and can be considered as different distributions, so we turn them into 0)
        return acc * p_vals

    # Process both datasets
    datasets = [("uncorrected", data_uncorrected), ("corrected", data_corrected)]

    # counters used to plot the data in the appropiate subplot
    contador = 0
    contador_plot = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    for dataset_idx, (dataset_name, dataset) in enumerate(datasets):
        for var_idx, (var_name, var_config) in enumerate(variables.items()):
            for lead_idx in range(2):  # 1 and 2 week lead times
                # Get climatology from input parameter

                climatology = climatology_datasets[lead_idx][var_name]  # [lead_idx + 1]
                climatology = climatology.mean(dim="number").squeeze()

                # Prepare observations
                obs_data = era_weeks[lead_idx].drop_vars(var_config["obs_vars"])
                if var_config["temp_convert"]:
                    obs_data = obs_data - 273.15

                # Get forecast data
                forecast_data = (
                    dataset[var_name]
                    .where(dataset["step.days"] == (7 + 7 * lead_idx), drop=True)
                    .mean(dim="number")
                )

                # Calculate ACC
                acc = calculate_acc(forecast_data, obs_data, climatology, var_name)

                # Plot results

                ax_idx = contador_plot[contador]
                contador += 1
                cs = axs[ax_idx].contourf(
                    dataset["lon"],
                    dataset["lat"],
                    acc,
                    transform=ccrs.PlateCarree(),
                    levels=acc_levels,
                    cmap="PiYG",
                )
                setup_map(axs[ax_idx])

    # Add titles and labels
    axs[0].set_title(lead_labels[0])
    axs[1].set_title(lead_labels[1])
    axs[2].set_title(lead_labels[0])
    axs[3].set_title(lead_labels[1])
    # Add subplot labels

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    positions = [0, 4, 8, 2, 6, 10]
    for i, pos in enumerate(positions):
        axs[pos].text(
            103.5,
            10.5,
            labels[i],
            transform=ccrs.Geodetic(),
            color="k",
            fontsize=fontsize,
        )

    # Add dataset labels
    axs[0].text(
        105.4,
        12.9,
        "ACC raw forecast",
        transform=ccrs.Geodetic(),
        color="k",
        fontsize=fontsize + 1,
    )
    axs[2].text(
        105.4,
        12.9,
        "ACC calibrated forecast",
        transform=ccrs.Geodetic(),
        color="k",
        fontsize=fontsize + 1,
    )

    # Add colorbar
    cbar = fig.colorbar(
        cs, fig.add_axes([0.92, 0.08, 0.025, 0.8]), orientation="vertical"
    )
    cbar.ax.tick_params(labelsize=fontsize - 2)

    return fig


observations = path_e + "/Daily_mean_ERA5_sh_rh.nc"  #

# Calculate climatology for all variables
variables_to_process = ["t2m", "r", "tp"]
ensemble_data = data_corr_s.sel(step="7.days")

climatology1 = calculate_climatology_multi_var(
    2004, 2020, ensemble_data, observations, variables_to_process
)
ensemble_data = data_corr_s.sel(step="14.days")
climatology2 = calculate_climatology_multi_var(
    2004, 2020, ensemble_data, observations, variables_to_process
)


# Measure climatology for the two weeks
climatology_data = [
    climatology1.sel(lon=slice(104.25, 107.5), lat=slice(12, 8.5)),
    climatology2.sel(lon=slice(104.25, 107.5), lat=slice(12, 8.5)),
]
# Call the function

fig = calculate_acc_maps(
    data_corr_s, data_uncorr_s, era_week1, era_week2, climatology_data
)  # plt.savefig("Figures_paper/Fig_3_Aggregation.tiff", dpi=600)
