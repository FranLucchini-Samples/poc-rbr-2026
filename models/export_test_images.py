"""Export the held-out test-set leaves as viewable PNGs.

The source .tif bands are tiny single-channel 16-bit rasters, not directly
viewable -- this builds a false-color composite (band 3,2,1 -> R,G,B, per the
paper's blue/green/red/red-edge/NIR band order) for each test leaf/stage,
resized to standard HD (1280x720).

Run with: uv run python models/export_test_images.py
"""

import re
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "DATASET"
OUTPUT_DIR = ROOT / "models" / "test_images"
SPECIES = ["Avocado", "Olive", "Vineyard"]
PATTERN = re.compile(r"leaf(\d+)d(\d+)_(\d+)\.tif")
TEST_SIZE = 0.15  # must match precompute.py / retraining.ipynb to reproduce the same held-out leaves
HD_SIZE = (1280, 720)  # width, height


def build_samples() -> list[dict]:
    leaf_counts = {}
    for sp in SPECIES:
        s = sp.lower()
        leaf_counts[sp] = sio.loadmat(DATA_ROOT / sp / f"FMC_{s}.mat")[f"FMC_d_{s}"].shape[0]

    samples = []
    for sp in SPECIES:
        img_dir = DATA_ROOT / sp / "Multispectral Images"
        by_key = {}
        for p in img_dir.glob("*.tif"):
            m = PATTERN.match(p.name)
            if not m:
                continue
            leaf, stage, band = int(m.group(1)), int(m.group(2)), int(m.group(3))
            by_key.setdefault((leaf, stage), {})[band] = p
        for (leaf, stage), bands in by_key.items():
            if set(bands) != {1, 2, 3, 4, 5}:
                continue  # incomplete band set -- skip defensively
            if leaf >= leaf_counts[sp]:
                continue  # no label row for this leaf -- skip defensively
            samples.append({
                "species": sp, "leaf": leaf, "stage": stage,
                "band_paths": [bands[b] for b in range(1, 6)],
            })
    return samples


def test_split(samples: list[dict]) -> np.ndarray:
    groups = np.array([f"{s['species']}_{s['leaf']}" for s in samples])
    all_idx = np.arange(len(samples))
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=0)
    _, test_idx = next(splitter.split(all_idx, groups=groups))
    return test_idx


def false_color_png(band_paths: list[Path], size: tuple[int, int]) -> Image.Image:
    r = np.array(Image.open(band_paths[2]).resize(size, Image.BILINEAR), dtype=np.float32)
    g = np.array(Image.open(band_paths[1]).resize(size, Image.BILINEAR), dtype=np.float32)
    b = np.array(Image.open(band_paths[0]).resize(size, Image.BILINEAR), dtype=np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    stretched = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return Image.fromarray((stretched * 255).astype(np.uint8))


if __name__ == "__main__":
    samples = build_samples()
    test_idx = test_split(samples)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for idx in test_idx:
        s = samples[idx]
        img = false_color_png(s["band_paths"], HD_SIZE)
        out_path = OUTPUT_DIR / f"{s['species']}_leaf{s['leaf']:03d}_stage{s['stage']}.png"
        img.save(out_path)

    total_bytes = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.png"))
    print(f"wrote {len(test_idx)} PNGs ({HD_SIZE[0]}x{HD_SIZE[1]}) to {OUTPUT_DIR.resolve()}")
    print(f"total size: {total_bytes / 1e6:.1f} MB")
