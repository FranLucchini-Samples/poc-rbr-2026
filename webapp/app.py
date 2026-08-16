"""A small FastAPI app that serves a paddle-style image gallery.

Users flick through the drone RGB tiles in a coverflow-like "paddle" and pick
one. The backend lists the images, serves lightweight cached thumbnails (the
source JPGs are large), streams the full-resolution image on demand, and
records the current selection in memory.

Run it with:

    uv run uvicorn webapp.app:app --reload

then open http://127.0.0.1:8000
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from starlette.requests import Request

# --- Paths -----------------------------------------------------------------

WEBAPP_DIR = Path(__file__).resolve().parent

# Folder holding the images to browse. Override with GALLERY_DIR if you want to
# point the gallery somewhere else.
DEFAULT_IMAGE_DIR = WEBAPP_DIR / "gallery" / "test_images"
IMAGE_DIR = Path(os.environ.get("GALLERY_DIR", DEFAULT_IMAGE_DIR)).resolve()

# Model predictions, keyed by RGB image filename (see results.json's "filename").
RESULTS_PATH = WEBAPP_DIR / "results.json"

# Per-species trait distributions/tiers (very low..very high), used to judge
# whether a predicted value is high or low relative to that species' norms.
LABEL_TIERS_PATH = WEBAPP_DIR / "label_tiers.json"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
THUMB_MAX = (480, 480)

templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

app = FastAPI(title="Paddle Gallery")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")


# --- In-memory state -------------------------------------------------------

# Latest selection made by a user. Fine for a PoC / single-user demo.
_selection: dict[str, str | None] = {"name": None}


# --- Helpers ---------------------------------------------------------------


def list_image_names() -> list[str]:
    """Return the sorted file names of gallery images."""
    if not IMAGE_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES
    )


def dataset_label() -> str:
    """A human-friendly name for the gallery folder, e.g. for the subtitle."""
    return IMAGE_DIR.parent.name.replace("-", " ").replace("_", " ").title()


def resolve_image(name: str) -> Path:
    """Resolve an image name to a path, guarding against traversal."""
    # Reject anything that tries to escape the gallery directory.
    candidate = (IMAGE_DIR / name).resolve()
    if IMAGE_DIR not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid image name")
    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


# Human-readable label/unit for each trait, fuel moisture first since it's
# the one that drives the leaf status bar (see leaf_status).
TRAIT_LABELS = {
    "fuel_moisture": ("Fuel moisture content", "%"),
    "chlorophyll": ("Chlorophyll", "SPAD"),
    "nitrogen": ("Nitrogen", "mg/g"),
    "weight": ("Weight", "g"),
}

# Our trait keys don't always match results.json's "predicted" keys.
PREDICTION_KEYS = {
    "fuel_moisture": "fmc",
    "chlorophyll": "chlorophyll",
    "nitrogen": "nitrogen",
    "weight": "weight",
}


def _load_samples() -> dict[str, dict]:
    """Map RGB image filename -> {"species", "predicted", "actual"} from results.json."""
    data = json.loads(RESULTS_PATH.read_text())
    return {
        sample["filename"]: {
            "species": sample["species"],
            "predicted": sample["predicted"],
            "actual": sample["actual"],
        }
        for sample in data["samples"]
    }


SAMPLES = _load_samples()
LABEL_TIERS = json.loads(LABEL_TIERS_PATH.read_text())

# Below this, predicted and actual are considered equal (avoids a jittery
# arrow from float noise when they're basically the same value).
DIFF_EPSILON = 0.05


def real_traits(name: str) -> list[dict[str, str | float]] | None:
    """Model-predicted trait values for this image, plus the actual measured
    value and whether the prediction ran higher or lower than it."""
    sample = SAMPLES.get(name)
    if sample is None:
        return None
    predicted = sample["predicted"]
    actual = sample["actual"]
    traits = []
    for key, (label, unit) in TRAIT_LABELS.items():
        pred_key = PREDICTION_KEYS[key]
        value = round(predicted[pred_key], 1)
        actual_value = round(actual[pred_key], 1)
        diff = value - actual_value
        if abs(diff) < DIFF_EPSILON:
            direction = "same"
        elif diff > 0:
            direction = "higher"
        else:
            direction = "lower"
        traits.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "value": value,
                "actual": actual_value,
                "direction": direction,
            }
        )
    return traits


# The 0-100 dryness score is split into four equal 25-point bands, from
# healthiest to most stressed.
STATUS_TIERS = [
    (25, "Ok", "✅"),
    (50, "Warning", "⚠️"),
    (75, "Danger", "🚨"),
    (100, "Fire", "🔥"),
]


def leaf_status(name: str) -> dict[str, float | str] | None:
    """Classify fuel moisture content against this species' own distribution.

    label_tiers.json gives each species its own observed fmc range (moisture
    varies a lot between e.g. Avocado and Vineyard leaves), so "dry" for one
    species isn't the same raw value as "dry" for another. Very high moisture
    relative to the species' range means healthy (score near 0); very low
    moisture means dry (score near 100).
    """
    sample = SAMPLES.get(name)
    if sample is None:
        return None
    species_fmc = LABEL_TIERS[sample["species"]]["fmc"]
    low, high = species_fmc["min"], species_fmc["max"]
    fmc = sample["predicted"]["fmc"]
    normalized = (fmc - low) / (high - low) if high > low else 0.0
    normalized = min(1.0, max(0.0, normalized))
    score = round((1 - normalized) * 100, 1)

    for max_score, label, emoji in STATUS_TIERS:
        if score <= max_score:
            break

    return {"score": score, "label": label, "emoji": emoji}


@lru_cache(maxsize=512)
def _thumbnail_bytes(name: str, mtime_ns: int) -> bytes:
    """Build (and cache) a JPEG thumbnail for an image.

    ``mtime_ns`` is part of the cache key so edited files get a fresh thumbnail.
    """
    path = resolve_image(name)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMB_MAX, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


# --- Models ----------------------------------------------------------------


class Selection(BaseModel):
    name: str


# --- Routes ----------------------------------------------------------------


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "image_dir": str(IMAGE_DIR),
            "dataset_label": dataset_label(),
            "count": len(list_image_names()),
        },
    )


@app.get("/api/images")
def api_images():
    names = list_image_names()
    return {
        "count": len(names),
        "images": [
            {
                "name": name,
                "thumb": f"/thumbs/{name}",
                "full": f"/images/{name}",
            }
            for name in names
        ],
    }


@app.get("/thumbs/{name}")
def thumb(name: str):
    path = resolve_image(name)
    data = _thumbnail_bytes(name, path.stat().st_mtime_ns)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/images/{name}")
def full_image(name: str):
    path = resolve_image(name)
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/detail")
def detail(request: Request):
    name = _selection["name"]
    traits = real_traits(name) if name else None
    status = leaf_status(name) if name else None
    return templates.TemplateResponse(
        request,
        "detail.html",
        {"name": name, "traits": traits, "status": status},
    )


@app.get("/api/selection")
def get_selection():
    return {"name": _selection["name"]}


@app.post("/api/selection")
def set_selection(selection: Selection):
    # Validate the name points to a real image before storing it.
    resolve_image(selection.name)
    _selection["name"] = selection.name
    return JSONResponse({"ok": True, "name": selection.name})
