"""Run the held-out test split through the latest saved checkpoint, and write
the results to models/checkpoints/results.json.

Run with: uv run python models/precompute.py
"""

import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "DATASET"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
RESULTS_PATH = CHECKPOINTS_DIR / "results.json"
SPECIES = ["Avocado", "Olive", "Vineyard"]
PATTERN = re.compile(r"leaf(\d+)d(\d+)_(\d+)\.tif")
TEST_SIZE = 0.15  # must match retraining.ipynb to reproduce the same held-out leaves


def latest_checkpoint() -> Path:
    checkpoints = sorted(
        p for p in CHECKPOINTS_DIR.glob("*.pt") if p.name != "results.json"
    )
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints found in {CHECKPOINTS_DIR}")
    return checkpoints[-1]


def load_labels(species: str) -> dict:
    s = species.lower()
    fmc = sio.loadmat(DATA_ROOT / species / f"FMC_{s}.mat")[f"FMC_d_{s}"]
    chl = sio.loadmat(DATA_ROOT / species / f"Chlorophyll_{s}.mat")[f"chlorophyll_{s}"]
    nit = sio.loadmat(DATA_ROOT / species / f"Nitrogen_{s}.mat")[f"nitrogen_{s}"]
    weight_file = DATA_ROOT / species / f"Weight_{species}.mat"  # avocado's filename capitalizes species
    if not weight_file.exists():
        weight_file = DATA_ROOT / species / f"Weight_{s}.mat"
    wgt = sio.loadmat(weight_file)[f"weight_{s}"]
    return {"fmc": fmc, "chlorophyll": chl, "nitrogen": nit, "weight": wgt}


def build_samples(traits: list[str]) -> list[dict]:
    labels_by_species = {sp: load_labels(sp) for sp in SPECIES}
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
            labels = labels_by_species[sp]
            if leaf >= labels["fmc"].shape[0]:
                continue  # no label row for this leaf -- skip defensively
            target = np.array([labels[t][leaf, stage] for t in traits], dtype=np.float32)
            samples.append({
                "species": sp, "leaf": leaf, "stage": stage,
                "band_paths": [bands[b] for b in range(1, 6)], "target": target,
            })
    return samples


def test_split(samples: list[dict]) -> np.ndarray:
    groups = np.array([f"{s['species']}_{s['leaf']}" for s in samples])
    all_idx = np.arange(len(samples))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=0)
    _, test_idx = next(test_splitter.split(all_idx, groups=groups))
    return test_idx


def load_model_input(band_paths: list[Path], img_size: int) -> np.ndarray:
    bands = [np.array(Image.open(p).resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)
              for p in band_paths]
    return np.stack(bands, axis=0) / 65535.0


def build_model(checkpoint: dict) -> nn.Module:
    model = resnet18(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(checkpoint["in_channels"], old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["traits"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_test_set(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    traits = checkpoint["traits"]
    img_size = checkpoint["img_size"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    model = build_model(checkpoint)
    samples = build_samples(traits)
    test_idx = test_split(samples)

    X_test = np.stack([load_model_input(samples[i]["band_paths"], img_size) for i in test_idx])
    y_test = np.stack([samples[i]["target"] for i in test_idx])

    with torch.no_grad():
        preds_z = model(torch.from_numpy(X_test)).numpy()
    preds = preds_z * y_std + y_mean

    metrics = {
        t: {
            "r2": float(r2_score(y_test[:, i], preds[:, i])),
            "mae": float(mean_absolute_error(y_test[:, i], preds[:, i])),
        }
        for i, t in enumerate(traits)
    }

    sample_results = [
        {
            "species": samples[idx]["species"],
            "leaf": samples[idx]["leaf"],
            "stage": samples[idx]["stage"],
            "images": [p.name for p in samples[idx]["band_paths"]],
            "actual": {t: float(y_test[pos, i]) for i, t in enumerate(traits)},
            "predicted": {t: float(preds[pos, i]) for i, t in enumerate(traits)},
        }
        for pos, idx in enumerate(test_idx)
    ]

    return {
        "checkpoint": checkpoint_path.name,
        "n_samples": len(test_idx),
        "metrics": metrics,
        "samples": sample_results,
    }


if __name__ == "__main__":
    checkpoint_path = latest_checkpoint()
    print(f"running test set through {checkpoint_path.name} ...")
    results = run_test_set(checkpoint_path)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_PATH.resolve()} ({RESULTS_PATH.stat().st_size / 1e3:.1f} KB)")
    for trait, m in results["metrics"].items():
        print(f"  {trait:12s} r2={m['r2']:.4f}  mae={m['mae']:.4f}")
