"""
Shared utilities for synthetic dataset generation.

The helpers here remain dependency-light to keep the generator script focused on
geometry authoring and labelling logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    """
    Load an STL mesh using trimesh.

    The returned mesh is ensured to be watertight where possible by repairing
    normals, although the generator already strives to emit closed solids.
    """
    mesh = trimesh.load_mesh(Path(path), file_type="stl")
    if not isinstance(mesh, trimesh.Trimesh):
        # Trimesh may return a Scene for multipart files; merge into a single mesh.
        mesh = trimesh.util.concatenate(mesh.dump())
    if not mesh.is_watertight:
        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
    mesh.rezero()
    return mesh


def compute_bbox(mesh: trimesh.Trimesh) -> list[float]:
    """
    Compute an axis-aligned bounding box as a flat list [x, y, z].

    The values correspond to the span along each axis (max - min) in millimetres,
    matching CadQuery / CAD default units.
    """
    bounds = mesh.bounds  # shape (2, 3)
    extents = bounds[1] - bounds[0]
    return [float(axis_len) for axis_len in extents]


def ensure_dir(path: Path | str) -> Path:
    """
    Create the directory if it does not yet exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_meta(meta_path: Path | str, meta: dict) -> None:
    """
    Persist the metadata dictionary to JSON with nice formatting for diffing.
    """
    meta_path = Path(meta_path)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def sample_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """
    Convenience wrapper returning a float sampled uniformly between low and high.
    """
    return float(rng.uniform(low, high))


def rolling_seed_sequence(seed: int, count: int) -> Iterable[int]:
    """
    Produce a deterministic sequence of seeds derived from an initial seed.
    """
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield int(rng.integers(0, np.iinfo(np.int32).max))


__all__ = [
    "compute_bbox",
    "ensure_dir",
    "load_mesh",
    "rolling_seed_sequence",
    "sample_uniform",
    "write_meta",
]

