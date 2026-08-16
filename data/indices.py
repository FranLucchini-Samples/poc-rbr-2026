"""Spectral indices computable from this dataset's four bands (RED, GRE/green,
NIR, REG/red edge), per the formulas listed at
https://eos.com/blog/vegetation-indices/.

That article also covers ARVI, EVI, VARI and SIPI (need a BLUE band), NBR and
NDSI (need a SWIR band), the NDSI/NDVI/NDWI "index stack" (needs NDSI), and
LAI (an allometric field measurement, not a band formula) — none of those are
computable with this camera's four bands, so they're omitted here.
"""

import numpy as np

SAVI_L = 0.5


def ndvi(bands):
    red, nir = bands["RED"], bands["NIR"]
    return (nir - red) / (nir + red)


def ndwi(bands):
    green, nir = bands["GRE"], bands["NIR"]
    return (green - nir) / (green + nir)


def ndre(bands):
    nir, red_edge = bands["NIR"], bands["REG"]
    return (nir - red_edge) / (nir + red_edge)


def gndvi(bands):
    green, nir = bands["GRE"], bands["NIR"]
    return (nir - green) / (nir + green)


def reci(bands):
    red, nir = bands["RED"], bands["NIR"]
    return (nir / red) - 1


def gci(bands):
    green, nir = bands["GRE"], bands["NIR"]
    return (nir / green) - 1


def savi(bands):
    red, nir = bands["RED"], bands["NIR"]
    return ((nir - red) / (nir + red + SAVI_L)) * (1 + SAVI_L)


def osavi(bands):
    red, nir = bands["RED"], bands["NIR"]
    return (nir - red) / (nir + red + 0.16)


def msavi(bands):
    red, nir = bands["RED"], bands["NIR"]
    discriminant = np.clip((2 * nir + 1) ** 2 - 8 * (nir - red), 0, None)
    return (2 * nir + 1 - np.sqrt(discriminant)) / 2


def ndwi_ndvi_masked(bands, th=0.1):
    """NDWI with pixels where NDVI > th is kept; the rest is set to NaN."""
    ndvi_vals = ndvi(bands)
    ndwi_vals = ndwi(bands)
    return ndwi_vals.where(ndvi_vals > th)


# Each spec: index function, colormap, and a fixed (vmin, vmax) for
# normalized-difference-style indices bounded to [-1, 1]; None lets
# matplotlib scale to the data for the unbounded ratio indices.
INDEX_SPECS = {
    "NDVI": (ndvi, "RdYlGn", (-1, 1)),
    "NDWI": (ndwi, "RdYlBu", (-1, 1)),
    "NDRE": (ndre, "RdYlGn", (-1, 1)),
    "GNDVI": (gndvi, "RdYlGn", (-1, 1)),
    "ReCI": (reci, "YlGn", None),
    "GCI": (gci, "YlGn", None),
    "SAVI": (savi, "RdYlGn", (-1, 1)),
    "OSAVI": (osavi, "RdYlGn", (-1, 1)),
    "MSAVI": (msavi, "RdYlGn", (-1, 1)),
    "NDWI (NDVI≤1)": (ndwi_ndvi_masked, "RdYlBu", (-1, 1)),
}
