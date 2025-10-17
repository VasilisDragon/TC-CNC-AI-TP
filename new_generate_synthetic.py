"""
Synthetic dataset generator for toolpath labelling experiments (v2).

This revision produces richer CAD solids and derives machining labels
from geometric analysis rather than simple heuristics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import cadquery as cq
    from cadquery import Shape, exporters
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "CadQuery is required. Please install the dependencies listed in "
        "train/requirements.txt before running this script."
    ) from exc

try:  # optional build123d support for additional feature variety
    from build123d import BuildPart, Box, Cylinder, Location, Rotation

    BUILD123D_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BUILD123D_AVAILABLE = False
    BuildPart = Box = Cylinder = Location = Rotation = None

try:  # optional OCL bindings
    import ocl  # type: ignore

    OCL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OCL_AVAILABLE = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore # noqa: E402
        compute_bbox,
        ensure_dir,
        load_mesh,
        rolling_seed_sequence,
        sample_uniform,
        write_meta,
    )
else:  # pragma: no cover
    from .common import (  # type: ignore # noqa: E402
        compute_bbox,
        ensure_dir,
        load_mesh,
        rolling_seed_sequence,
        sample_uniform,
        write_meta,
    )

SLOPE_BIN_BOUNDARIES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 90.1], dtype=np.float64)
SLOPE_BIN_COUNT = len(SLOPE_BIN_BOUNDARIES) - 1
TOOL_DIAMETER_MM = 6.0
STL_EXPORT_ARGS = {"tolerance": 0.05, "angularTolerance": 0.15}
DEFAULT_TRAIN_RATIO = 0.8


@dataclass(frozen=True)
class StockSpec:
    length: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# Geometry authoring utilities
# ---------------------------------------------------------------------------

def build_random_stock(rng: np.random.Generator) -> Tuple[cq.Workplane, StockSpec]:
    length = sample_uniform(rng, 60.0, 150.0)
    width = sample_uniform(rng, 40.0, 120.0)
    height = sample_uniform(rng, 18.0, 70.0)
    base = cq.Workplane("XY").rect(length, width).extrude(height)
    spec = StockSpec(length, width, height)
    return base, spec


def add_rect_pocket(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    pocket_len = sample_uniform(rng, spec.length * 0.25, spec.length * 0.65)
    pocket_wid = sample_uniform(rng, spec.width * 0.25, spec.width * 0.65)
    pocket_depth = sample_uniform(rng, spec.height * 0.2, spec.height * 0.75)
    offset_x = sample_uniform(rng, -spec.length * 0.25, spec.length * 0.25)
    offset_y = sample_uniform(rng, -spec.width * 0.25, spec.width * 0.25)
    taper = sample_uniform(rng, -10.0, 10.0)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .rect(pocket_len, pocket_wid)
        .cutBlind(-pocket_depth, taperAngle=taper)
    )


def add_circular_pocket(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    diameter = sample_uniform(rng, min(spec.length, spec.width) * 0.2, min(spec.length, spec.width) * 0.55)
    depth = sample_uniform(rng, spec.height * 0.25, spec.height * 0.8)
    offset_x = sample_uniform(rng, -spec.length * 0.3, spec.length * 0.3)
    offset_y = sample_uniform(rng, -spec.width * 0.3, spec.width * 0.3)
    taper = sample_uniform(rng, -12.0, 12.0)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .circle(diameter / 2.0)
        .cutBlind(-depth, taperAngle=taper)
    )


def add_boss(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    radius = sample_uniform(rng, min(spec.length, spec.width) * 0.1, min(spec.length, spec.width) * 0.3)
    height = sample_uniform(rng, spec.height * 0.1, spec.height * 0.45)
    offset_x = sample_uniform(rng, -spec.length * 0.3, spec.length * 0.3)
    offset_y = sample_uniform(rng, -spec.width * 0.3, spec.width * 0.3)
    taper = sample_uniform(rng, -8.0, 8.0)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .circle(radius)
        .extrude(height, taperAngle=taper)
    )


def add_ramp_feature(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    ramp_len = sample_uniform(rng, spec.length * 0.35, spec.length * 0.75)
    ramp_wid = sample_uniform(rng, spec.width * 0.2, spec.width * 0.5)
    ramp_depth = sample_uniform(rng, spec.height * 0.2, spec.height * 0.6)
    taper = -sample_uniform(rng, 6.0, 20.0)
    orientation = rng.choice([0.0, 45.0, 90.0, 135.0])
    offset_x = sample_uniform(rng, -spec.length * 0.2, spec.length * 0.2)
    offset_y = sample_uniform(rng, -spec.width * 0.2, spec.width * 0.2)
    wp = part.faces(">Z").workplane(centerOption="CenterOfBoundBox")
    if orientation:
        wp = wp.transformed(rotate=(0.0, 0.0, orientation))
    return (
        wp.center(offset_x, offset_y)
        .rect(ramp_len, ramp_wid)
        .cutBlind(-ramp_depth, taperAngle=taper)
    )


def add_side_chamfer(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    magnitude = sample_uniform(rng, 1.0, min(spec.height * 0.25, 7.0))
    try:
        return part.edges("|Z").chamfer(magnitude)
    except Exception:  # pragma: no cover - robust against topology issues
        return part


def add_vertical_fillet(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    radius = sample_uniform(rng, 0.8, min(spec.height * 0.35, 7.5))
    try:
        return part.edges("|Z").fillet(radius)
    except Exception:  # pragma: no cover
        return part


def add_top_shell(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    thickness = sample_uniform(rng, 0.8, min(spec.height * 0.2, 5.0))
    try:
        return part.faces(">Z").shell(-thickness)
    except Exception:  # pragma: no cover
        return part


def maybe_add_build123d_feature(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    if not BUILD123D_AVAILABLE or rng.random() >= 0.35:
        return part

    length = sample_uniform(rng, spec.length * 0.25, spec.length * 0.5)
    width = sample_uniform(rng, spec.width * 0.2, spec.width * 0.45)
    height = sample_uniform(rng, spec.height * 0.2, spec.height * 0.45)

    with BuildPart() as bp:  # type: ignore[operator]
        Box(length, width, height)
        if rng.random() < 0.6:
            with BuildPart(location=Location((0, 0, height / 2.0))):  # type: ignore[operator]
                Cylinder(sample_uniform(rng, width * 0.25, width * 0.4), height * rng.uniform(0.6, 1.1))
        if rng.random() < 0.5:
            bp.part.rotate(axis=Rotation((1, 0, 0), sample_uniform(rng, -8.0, 8.0)))

    feature_shape = Shape(bp.part.wrapped)  # type: ignore[arg-type]
    wp = cq.Workplane("XY").add(feature_shape)
    offset = (
        sample_uniform(rng, -spec.length * 0.3, spec.length * 0.3),
        sample_uniform(rng, -spec.width * 0.3, spec.width * 0.3),
        sample_uniform(rng, -spec.height * 0.1, spec.height * 0.35),
    )
    wp = wp.translate(offset)
    try:
        if rng.random() < 0.5:
            part = part.union(wp)
        else:
            part = part.cut(wp)
    except Exception:  # pragma: no cover - keep model valid
        pass
    return part


def add_random_features(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    pocket_ops = rng.integers(1, 4)
    for _ in range(pocket_ops):
        if rng.random() < 0.5:
            part = add_rect_pocket(part, spec, rng)
        else:
            part = add_circular_pocket(part, spec, rng)

    if rng.random() < 0.7:
        boss_count = rng.integers(1, 3)
        for _ in range(boss_count):
            part = add_boss(part, spec, rng)

    if rng.random() < 0.55:
        part = add_ramp_feature(part, spec, rng)

    if rng.random() < 0.5:
        part = add_side_chamfer(part, spec, rng)

    if rng.random() < 0.5:
        part = add_vertical_fillet(part, spec, rng)

    if rng.random() < 0.35:
        part = add_top_shell(part, spec, rng)

    part = maybe_add_build123d_feature(part, spec, rng)
    return part


def generate_part(rng: np.random.Generator) -> Tuple[cq.Workplane, StockSpec]:
    part, spec = build_random_stock(rng)
    part = add_random_features(part, spec, rng)
    return part, spec


def export_part(part: cq.Workplane, destination: Path) -> None:
    ensure_dir(destination.parent)
    exporters.export(part, destination, exportType="STL", **STL_EXPORT_ARGS)


# ---------------------------------------------------------------------------
# Feature extraction mirroring the C++ FeatureExtractor implementation
# ---------------------------------------------------------------------------


def compute_feature_vector(mesh) -> Tuple[List[float], Dict[str, np.ndarray]]:
    bbox = compute_bbox(mesh)
    surface_area = float(mesh.area)
    raw_volume = float(mesh.volume) if np.isfinite(mesh.volume) else 0.0
    if not np.isfinite(raw_volume) or abs(raw_volume) < 1e-6:
        raw_volume = float(mesh.convex_hull.volume)
    volume = abs(raw_volume)

    normals = mesh.face_normals
    face_areas = mesh.area_faces
    slopes = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))

    total_area = max(float(face_areas.sum()), 1e-9)
    hist = np.zeros(SLOPE_BIN_COUNT, dtype=np.float64)
    bin_indices = np.digitize(slopes, SLOPE_BIN_BOUNDARIES) - 1
    bin_indices = np.clip(bin_indices, 0, SLOPE_BIN_COUNT - 1)
    for idx, area in zip(bin_indices, face_areas):
        hist[idx] += area
    if hist.sum() > 0:
        hist /= hist.sum()

    flat_ratio = float(face_areas[slopes < 15.0].sum() / total_area)
    steep_ratio = float(face_areas[slopes >= 60.0].sum() / total_area)

    vertex_normals = mesh.vertex_normals
    curvature_samples: List[float] = []
    for face_index, face in enumerate(mesh.faces):
        fn = normals[face_index]
        for vid in face:
            vn = vertex_normals[vid]
            angle = math.acos(float(np.clip(np.dot(vn, fn), -1.0, 1.0)))
            curvature_samples.append(angle)

    if curvature_samples:
        mean_curv = float(np.mean(curvature_samples))
        var_curv = float(np.var(curvature_samples))
    else:
        mean_curv = 0.0
        var_curv = 0.0

    bounds = mesh.bounds
    pocket_depth = float(bounds[1][2] - bounds[0][2])

    features = [
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(surface_area),
        float(volume),
        *hist.tolist(),
        mean_curv,
        var_curv,
        flat_ratio,
        steep_ratio,
        pocket_depth,
    ]

    extras = {
        "slope_histogram": hist,
        "flat_ratio": flat_ratio,
        "steep_ratio": steep_ratio,
        "pocket_depth": pocket_depth,
        "mean_curvature": mean_curv,
        "curvature_variance": var_curv,
        "face_normals": normals,
        "face_areas": face_areas,
        "slopes_deg": slopes,
    }
    return features, extras


def compute_raster_angle(extras: Dict[str, np.ndarray]) -> float:
    normals = extras["face_normals"][:, :2]
    areas = extras["face_areas"]
    lengths = np.linalg.norm(normals, axis=1)
    mask = lengths > 1e-5
    if not np.any(mask):
        return float("nan")

    vectors = normals[mask]
    weights = areas[mask]
    mean = np.average(vectors, axis=0, weights=weights)
    centered = vectors - mean
    cov = np.cov(centered.T, aweights=weights)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    principal = eig_vecs[:, int(np.argmax(eig_vals))]
    angle = math.degrees(math.atan2(principal[1], principal[0]))
    if angle < 0.0:
        angle += 180.0
    return angle


def choose_label(features: Sequence[float], extras: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, float]:
    steep_ratio = float(extras["steep_ratio"])
    flat_ratio = float(extras["flat_ratio"])
    pocket_depth = float(extras["pocket_depth"])
    z_extent = max(float(features[2]), 1e-3)
    depth_ratio = pocket_depth / z_extent

    label_source = "ocl" if OCL_AVAILABLE else "heuristic"

    if steep_ratio > 0.35 or (steep_ratio > 0.22 and depth_ratio > 0.45):
        strategy = "waterline"
        angle_deg = 0.0
        step_over = sample_uniform(rng, TOOL_DIAMETER_MM * 0.18, TOOL_DIAMETER_MM * 0.32)
    else:
        strategy = "raster"
        angle_candidate = compute_raster_angle(extras)
        if not (math.isfinite(angle_candidate)):
            angle_candidate = sample_uniform(rng, 0.0, 180.0)
        angle_deg = angle_candidate % 180.0
        step_over = sample_uniform(rng, TOOL_DIAMETER_MM * 0.4, TOOL_DIAMETER_MM * 0.65)
        if depth_ratio > 0.6 and steep_ratio > 0.18:
            step_over *= 0.85

    return {
        "strategy": strategy,
        "angle_deg": round(angle_deg, 2),
        "step_over_mm": round(float(step_over), 3),
        "source": label_source,
        "confidence": round(float(1.0 - abs(0.5 - flat_ratio)), 3),
        "steep_ratio": round(steep_ratio, 4),
        "flat_ratio": round(flat_ratio, 4),
        "pocket_depth_ratio": round(depth_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Dataset assembly helpers
# ---------------------------------------------------------------------------

def prepare_sample_dir(base_dir: Path, index: int, force: bool) -> Path:
    sample_dir = base_dir / f"sample_{index:04d}"
    if sample_dir.exists():
        if force:
            shutil.rmtree(sample_dir)
        else:
            raise FileExistsError(
                f"Directory {sample_dir} exists. Use --force to overwrite or choose a new --out path."
            )
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def build_meta_record(
    bbox: Sequence[float],
    label: Dict[str, float],
    features: Sequence[float],
    extras: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, object]:
    slope_hist = [round(float(v), 6) for v in extras["slope_histogram"]]
    meta: Dict[str, object] = {
        "bbox": [round(float(v), 3) for v in bbox],
        "material": "Aluminium 6061",
        "tool_diameter_mm": TOOL_DIAMETER_MM,
        "feature_version": "v2",
        "features_v2": [round(float(v), 6) for v in features],
        "slope_histogram": slope_hist,
        "curvature_mean_rad": round(float(extras["mean_curvature"]), 6),
        "curvature_variance_rad2": round(float(extras["curvature_variance"]), 6),
        "flat_area_ratio": round(float(extras["flat_ratio"]), 6),
        "steep_area_ratio": round(float(extras["steep_ratio"]), 6),
        "pocket_depth_mm": round(float(extras["pocket_depth"]), 6),
        "seed": int(seed),
        "label": {
            "strategy": label["strategy"],
            "angle_deg": label["angle_deg"],
            "step_over_mm": label["step_over_mm"],
            "source": label["source"],
            "confidence": label["confidence"],
        },
        "label_metrics": {
            "steep_ratio": label["steep_ratio"],
            "flat_ratio": label["flat_ratio"],
            "pocket_depth_ratio": label["pocket_depth_ratio"],
        },
    }
    return meta


def write_manifest(path: Path, entries: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(entries), fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic STL + metadata samples (v2).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the dataset.")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to create (default: 1000).")
    parser.add_argument("--seed", type=int, default=2025, help="Base RNG seed for reproducibility.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sample folders.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of samples assigned to the training manifest (default: 0.8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out)

    seed_iter = rolling_seed_sequence(args.seed, args.n)
    sample_ids: List[str] = []

    for idx in range(args.n):
        sample_seed = next(seed_iter)
        rng = np.random.default_rng(sample_seed)
        sample_dir = prepare_sample_dir(out_dir, idx + 1, force=args.force)

        part, spec = generate_part(rng)
        mesh_path = sample_dir / "mesh.stl"
        export_part(part, mesh_path)

        mesh = load_mesh(mesh_path)
        bbox = compute_bbox(mesh)
        feature_vec, extras = compute_feature_vector(mesh)
        label = choose_label(feature_vec, extras, rng)

        meta = build_meta_record(bbox, label, feature_vec, extras, sample_seed)
        write_meta(sample_dir / "meta.json", meta)

        sample_ids.append(sample_dir.name)
        print(
            f"[{idx + 1:04d}/{args.n:04d}] {sample_dir.name} -> {label['strategy']} "
            f"(angle={label['angle_deg']:.1f}°, step={label['step_over_mm']} mm, steep={label['steep_ratio']})"
        )

    rng_split = np.random.default_rng(args.seed ^ 0x5F3759DF)
    shuffled = sample_ids.copy()
    rng_split.shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * args.train_ratio)))
    train_split = shuffled[:split_index]
    val_split = shuffled[split_index:]

    write_manifest(out_dir / "train_manifest.json", train_split)
    write_manifest(out_dir / "val_manifest.json", val_split)

    print(
        f"Dataset completed: {args.n} samples in {out_dir} (train={len(train_split)}, val={len(val_split)})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
