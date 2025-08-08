# Code to make fig 2 (bias between corrected and uncorrected data
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import os
data_corr_s = xr.open_dataset("data_corr_s_github.nc", decode_timedelta=True)
data_uncorr_s = xr.open_dataset("data_uncorr_s_github.nc", decode_timedelta=True)

era_week1 = xr.open_dataset("era_week1_south_github.nc")
era_week2 = xr.open_dataset("era_week2_south_github.nc")


# The 4 datasets should have the same lat and lon extension
def plot_bias_maps(data, suffix):
    """Plot bias maps"""

    # Levels for representing variables
    variables = {
        "T2m": {
            "levels": np.arange(-2, 2.5, 0.5),
            "unit": "°C",
            "label": "(a)",
            "var_key": "t2m",
        },
        "r": {
            "levels": np.arange(-10, 10.1, 1),
            "unit": "%",
            "label": "(b)",
            "var_key": "r",
        },
        "TP": {
            "levels": np.arange(-20, 21, 1),
            "unit": "mm",
            "label": "(c)",
            "var_key": "tp",
        },
    }

    lead_labels = ["1 week lead time", "2 week lead time"]
    era_weeks = [era_week1, era_week2]
    fontsize = 17

    # Retain data correspondant to rainy season (May- Oct)
    temp = data.where(
        (data["valid_time.month"] > 4) & (data["valid_time.month"] < 11), drop=True
    )

    def setup_map(ax):
        """Draw coastlines, boundaries and other geographical details for cartopy map"""
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
            106.5,
            11,
            "HCMC",
            transform=ccrs.Geodetic(),
            color="k",
            fontsize=fontsize - 7,
        )

    # Loop to obtain  variable images
    for var_name, config in variables.items():
        fig = plt.figure(figsize=(9.5 / 1.05, 6 / 1.5))

        for i in range(2):
            # Retain forecast data from a specific timestep
            forecast = temp.where(temp["step.days"] == (7 + 7 * i), drop=True).squeeze()
            dates = forecast["valid_time"]
            forecast = forecast.mean(
                dim=["number"]
            )  # measure mean state in forecast simulations

            # Retain observational data with the same dates as forecast
            obs = era_weeks[i].where(era_weeks[i].time.isin(dates), drop=True)

            # Calculate bias
            bias = (forecast - obs).mean(dim="time")

            # Create subplot and plot
            ax = fig.add_subplot(1, 3, i + 1, projection=ccrs.PlateCarree())
            cs = ax.contourf(
                bias.lon,
                bias.lat,
                bias[config["var_key"]].to_numpy(),
                levels=config["levels"],
                cmap="PiYG",
                extend="both",
            )

            # Setup map features
            setup_map(ax)

            # Add titles and labels
            if var_name == "T2m":
                ax.set_title(lead_labels[i], fontsize=fontsize - 8)

            if i == 0:
                ax.text(
                    103,
                    11,
                    config["label"],
                    transform=ccrs.Geodetic(),
                    color="k",
                    fontsize=fontsize - 3,
                )
                ax.set_yticks(np.arange(8.5, 11.5, 2))

                if var_name == "R":
                    ax.set_ylabel("Latitude (°)", fontsize=fontsize - 6, y=0.47)
                elif var_name == "TP":
                    ax.text(107.2, 7.7, "Longitude (°)", fontsize=fontsize - 6)

            if var_name == "TP":
                ax.set_xticks(np.arange(104.5, 107.5, 2))

        # Add colorbar
        cbar = fig.colorbar(
            cs, fig.add_axes([0.68, 0.21, 0.02, 0.55]), orientation="vertical"
        )
        cbar.set_label(config["unit"], rotation=0)

        # Save figure
        plt.savefig(f"bias_{var_name}_S_{suffix}.tiff", dpi=600)


def concatenate_images(suffix):
    """Concatenate bias images vertically."""

    # Load images
    images = [Image.open(f"bias_{var}_S_{suffix}.tiff") for var in ["T2m", "R", "TP"]]

    # Concatenate vertically
    def concat_v(im1, im2):
        dst = Image.new("RGB", (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    # Chain concatenation
    result = images[0]
    for img in images[1:]:
        result = concat_v(result, img)

    result.save(f"bias_{suffix}.tiff")


# Create bias map corrected and uncorrected data
plot_bias_maps(data_corr_s, "corr")
plot_bias_maps(data_uncorr_s, "uncorr")

# Concatenate images
concatenate_images("corr")
concatenate_images("uncorr")

#Deleting intermediate images
os.remove("bias_T2M_corr.tiff")
os.remove("bias_r_corr.tiff")
os.remove("bias_TP_corr.tiff")


os.remove("bias_T2M_corr.tiff")
os.remove("bias_r_corr.tiff")
os.remove("bias_TP_corr.tiff")
