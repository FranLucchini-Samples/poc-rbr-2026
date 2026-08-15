"""Fetch sample files from the Kaggle dataset "Mixed-use agricultural fields"
(marcosgabriel/mixeduse-agricultural-fields), preferring a local cache under
DATA_DIR over re-downloading via kagglehub's `path` argument.

The dataset holds AgEagle eBee Duet M drone captures of a Swiss crop field:
four single-band multispectral GeoTIFFs per tile (RED, GRE, NIR, REG) plus
separate high-resolution RGB visual photos. Only the files requested here
are fetched (one tile + one RGB sample), not the full ~2.7 GB dataset.
"""

from pathlib import Path

import kagglehub

DATASET = "marcosgabriel/mixeduse-agricultural-fields"
TILE_ID = "IMG_210204_095113_0000"
BANDS = ("RED", "GRE", "NIR", "REG")
RGB_SAMPLE = "rgb-images/IX-01-07922_0011_0001.JPG"
DATA_DIR = Path(__file__).parent / "mixeduse-agricultural-fields"


def fetch(relative_path: str) -> str:
    local_path = DATA_DIR / relative_path
    if local_path.exists():
        return str(local_path)
    return kagglehub.dataset_download(DATASET, path=relative_path)


def download_band(band: str) -> str:
    return fetch(f"multispectral-images/{band}/{TILE_ID}_{band}.tif")


def download_tile() -> dict[str, str]:
    return {band: download_band(band) for band in BANDS}


def download_rgb_sample() -> str:
    return fetch(RGB_SAMPLE)
