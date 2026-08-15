"""Read and visualize a sample tile from the "Mixed-use agricultural fields"
dataset. Dataset fetching lives in download.py, spectral index formulas live
in indices.py; this module only reads files and plots them.
"""

from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray
from PIL import Image
from rasterio.plot import show
from shapely.geometry import box

from .download import BANDS, TILE_ID, download_rgb_sample, download_tile
from .indices import INDEX_SPECS

PREVIEW_PATH = Path(__file__).parent / "tile_preview.png"
INDICES_PATH = Path(__file__).parent / "tile_indices.png"


def plot_bands(axes, band_paths):
    for ax, band in zip(axes, BANDS):
        with rasterio.open(band_paths[band]) as src:
            show(src, ax=ax, cmap="gray", title=band)
        ax.axis("off")


def plot_rgb_sample(ax, rgb_path):
    thumbnail = Image.open(rgb_path)
    thumbnail.thumbnail((1024, 1024))
    thumbnail = thumbnail.rotate(180)
    ax.imshow(thumbnail)
    ax.set_title("RGB sample (visual camera)")
    ax.axis("off")
    return thumbnail


def plot_edges(ax, rgb_thumbnail):
    gray = cv2.cvtColor(np.array(rgb_thumbnail.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ax.imshow(edges, cmap="gray")
    ax.set_title("Field edges (cv2.Canny)")
    ax.axis("off")


def plot_footprint(ax, band_paths):
    with rasterio.open(band_paths["RED"]) as src:
        footprint = gpd.GeoDataFrame({"geometry": [box(*src.bounds)]}, crs=src.crs)
    footprint.plot(ax=ax, edgecolor="crimson", facecolor="none", linewidth=2)
    crs_label = footprint.crs if footprint.crs else "no CRS in file (pixel bounds)"
    ax.set_title(f"Tile footprint\n{crs_label}")
    ax.ticklabel_format(useOffset=False, style="plain")


def plot_preview(band_paths, rgb_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Mixed-use agricultural fields — tile {TILE_ID}")

    plot_bands(axes[0], band_paths)
    rgb_thumbnail = plot_rgb_sample(axes[1, 0], rgb_path)
    plot_edges(axes[1, 1], rgb_thumbnail)
    plot_footprint(axes[1, 2], band_paths)
    axes[1, 3].axis("off")

    plt.tight_layout()
    fig.savefig(PREVIEW_PATH, dpi=150)
    print(f"Saved preview to {PREVIEW_PATH}")


def plot_indices(bands):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Spectral indices — tile {TILE_ID} (RED/GREEN/NIR/RED-EDGE only)")

    for ax, (name, (fn, cmap, value_range)) in zip(axes.flat, INDEX_SPECS.items()):
        result = fn(bands)
        vmin, vmax = value_range if value_range else (None, None)
        result.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=True)
        ax.set_title(f"{name} (mean={float(result.mean()):.2f})")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(INDICES_PATH, dpi=150)
    print(f"Saved indices to {INDICES_PATH}")


def main() -> None:
    band_paths = download_tile()
    rgb_path = download_rgb_sample()
    bands = {
        name: rioxarray.open_rasterio(path, masked=True).squeeze()
        for name, path in band_paths.items()
    }

    plot_preview(band_paths, rgb_path)
    plot_indices(bands)
    plt.show()


if __name__ == "__main__":
    main()
