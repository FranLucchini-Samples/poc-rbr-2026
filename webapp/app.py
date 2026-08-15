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

import hashlib
import io
import os
import random
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
PROJECT_ROOT = WEBAPP_DIR.parent

# Folder holding the images to browse. Override with GALLERY_DIR if you want to
# point the gallery somewhere else.
DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "mixeduse-agricultural-fields" / "rgb-images"
)
IMAGE_DIR = Path(os.environ.get("GALLERY_DIR", DEFAULT_IMAGE_DIR)).resolve()

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


def resolve_image(name: str) -> Path:
    """Resolve an image name to a path, guarding against traversal."""
    # Reject anything that tries to escape the gallery directory.
    candidate = (IMAGE_DIR / name).resolve()
    if IMAGE_DIR not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid image name")
    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


# Plausible ranges for each trait, used to generate mock PoC data. There is no
# real ground-truth measurement dataset behind this yet.
TRAIT_RANGES = {
    "fuel_moisture": ("Fuel moisture content", "%", 35.0, 75.0),
    "weight": ("Weight", "g", 120.0, 420.0),
    "chlorophyll": ("Chlorophyll", "SPAD", 20.0, 55.0),
    "nitrogen": ("Nitrogen", "mg/g", 12.0, 45.0),
}


@lru_cache(maxsize=512)
def mock_traits(name: str) -> list[dict[str, str | float]]:
    """Deterministic, per-image mock trait values (no real measurements exist).

    Seeded by the image name so the same image always shows the same values.
    """
    seed = hashlib.sha256(name.encode()).hexdigest()
    rng = random.Random(seed)
    return [
        {
            "key": key,
            "label": label,
            "unit": unit,
            "value": round(rng.uniform(low, high), 1),
        }
        for key, (label, unit, low, high) in TRAIT_RANGES.items()
    ]


# Traits that feed the leaf health score, i.e. the ones that actually track
# plant stress. Weight is a size/biomass measure, not a health signal.
HEALTH_TRAIT_KEYS = ("fuel_moisture", "chlorophyll", "nitrogen")


def leaf_status(traits: list[dict[str, str | float]]) -> dict[str, float | str]:
    """Roll fuel moisture, chlorophyll and nitrogen into a single stress score.

    0 means every underlying trait is at the healthy end of its range, 100
    means every trait is at the stressed end.
    """
    values = {t["key"]: t["value"] for t in traits}
    stress_components = []
    for key in HEALTH_TRAIT_KEYS:
        _, _, low, high = TRAIT_RANGES[key]
        normalized = (values[key] - low) / (high - low)
        stress_components.append(1 - normalized)
    score = round(sum(stress_components) / len(stress_components) * 100, 1)

    if score < 33:
        label = "Healthy"
    elif score < 66:
        label = "Stressed"
    else:
        label = "Dry"

    return {"score": score, "label": label}


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
    traits = mock_traits(name) if name else None
    status = leaf_status(traits) if traits else None
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
