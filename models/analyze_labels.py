"""Analyze per-species label distributions and derive a 5-tier standardized range
(very low / low / medium / high / very high) for each measured trait.

Tier cutoffs use the standard sigma-banding scheme: mean +/- 0.5 std and
mean +/- 1.5 std split the distribution into 5 bands, clipped to the observed
min/max so every tier's range is a real, reachable interval. Stats are computed
over all leaves x all 5 dehydration stages per species (there's no train/val/test
split here -- this describes the full label distribution, not model performance).

Run with: uv run python models/analyze_labels.py
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "DATASET"
OUTPUT_PATH = ROOT / "models" / "label_tiers.json"
SPECIES = ["Avocado", "Olive", "Vineyard"]
TRAITS = ["fmc", "chlorophyll", "nitrogen", "weight"]
TRAIT_UNITS = {"fmc": "%", "chlorophyll": "SPAD", "nitrogen": "mg/g", "weight": "g"}
TIER_LABELS = ["very low", "low", "medium", "high", "very high"]
SIGMA_CUTOFFS = [-1.5, -0.5, 0.5, 1.5]  # in units of std around the mean


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


def tier_edges(mean: float, std: float, data_min: float, data_max: float) -> list[float]:
    edges = [data_min]
    for k in SIGMA_CUTOFFS:
        cut = min(max(mean + k * std, data_min), data_max)
        edges.append(max(cut, edges[-1]))  # clip into range and keep non-decreasing
    edges.append(max(data_max, edges[-1]))
    return edges


def analyze(values: np.ndarray) -> dict:
    flat = values.astype(np.float64).ravel()
    mean, std = float(flat.mean()), float(flat.std())
    data_min, data_max = float(flat.min()), float(flat.max())
    edges = tier_edges(mean, std, data_min, data_max)

    tiers = [
        {"tier": i + 1, "label": label, "range": [round(edges[i], 3), round(edges[i + 1], 3)]}
        for i, label in enumerate(TIER_LABELS)
    ]

    return {
        "n": int(flat.size),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "median": round(float(np.median(flat)), 3),
        "min": round(data_min, 3),
        "max": round(data_max, 3),
        "tiers": tiers,
    }


if __name__ == "__main__":
    report = {}
    for sp in SPECIES:
        labels = load_labels(sp)
        report[sp] = {trait: analyze(labels[trait]) for trait in TRAITS}

    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT_PATH.resolve()}\n")

    for sp in SPECIES:
        print(sp)
        for trait in TRAITS:
            stats = report[sp][trait]
            unit = TRAIT_UNITS[trait]
            print(f"  {trait:12s} n={stats['n']:3d}  mean={stats['mean']:8.2f}  "
                  f"std={stats['std']:7.2f}  median={stats['median']:8.2f}  "
                  f"[{stats['min']:.2f}, {stats['max']:.2f}]")
            for t in stats["tiers"]:
                lo, hi = t["range"]
                print(f"      {t['tier']}  {t['label']:10s} [{lo:8.2f}, {hi:8.2f}] {unit}")
        print()
