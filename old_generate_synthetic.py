

Synthetic dataset generator for toolpath labelling experiments.

The generator produces small CAD solids using CadQuery, exports each sample as
an STL, and writes metadata describing machining strategy hints.


from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cadquery as cq
    from cadquery import exporters
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "CadQuery is required. Please install the dependencies listed in "
        "train/requirements.txt before running this script."
    ) from exc

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
else:
    from .common import (  # type: ignore # noqa: E402
        compute_bbox,
        ensure_dir,
        load_mesh,
        rolling_seed_sequence,
        sample_uniform,
        write_meta,
    )

TOOL_DIAMETER_MM = 6.0
STL_EXPORT_ARGS = {"tolerance": 0.08, "angularTolerance": 0.2}


@dataclass(frozen=True)
class StockSpec:
    length: float
    width: float
    height: float


def build_random_stock(rng: np.random.Generator) -> Tuple[cq.Workplane, StockSpec]:
    length = sample_uniform(rng, 60.0, 140.0)
    width = sample_uniform(rng, 40.0, 110.0)
    height = sample_uniform(rng, 15.0, 60.0)
    base = cq.Workplane("XY").rect(length, width).extrude(height)
    spec = StockSpec(length, width, height)
    return base, spec


def add_rect_pocket(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    pocket_len = sample_uniform(rng, spec.length * 0.25, spec.length * 0.6)
    pocket_wid = sample_uniform(rng, spec.width * 0.25, spec.width * 0.6)
    pocket_depth = sample_uniform(rng, spec.height * 0.2, spec.height * 0.7)
    offset_x = sample_uniform(rng, -spec.length * 0.25, spec.length * 0.25)
    offset_y = sample_uniform(rng, -spec.width * 0.25, spec.width * 0.25)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .rect(pocket_len, pocket_wid)
        .cutBlind(-pocket_depth)
    )


def add_circular_pocket(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    diameter = sample_uniform(rng, min(spec.length, spec.width) * 0.2, min(spec.length, spec.width) * 0.5)
    depth = sample_uniform(rng, spec.height * 0.25, spec.height * 0.8)
    offset_x = sample_uniform(rng, -spec.length * 0.3, spec.length * 0.3)
    offset_y = sample_uniform(rng, -spec.width * 0.3, spec.width * 0.3)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .circle(diameter / 2.0)
        .cutBlind(-depth)
    )


def add_boss(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    radius = sample_uniform(rng, min(spec.length, spec.width) * 0.1, min(spec.length, spec.width) * 0.25)
    height = sample_uniform(rng, spec.height * 0.1, spec.height * 0.4)
    offset_x = sample_uniform(rng, -spec.length * 0.3, spec.length * 0.3)
    offset_y = sample_uniform(rng, -spec.width * 0.3, spec.width * 0.3)
    return (
        part.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(offset_x, offset_y)
        .circle(radius)
        .extrude(height)
    )


def add_side_chamfer(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    magnitude = sample_uniform(rng, 1.0, min(spec.height * 0.25, 6.0))
    try:
        return part.edges("|Z").chamfer(magnitude)
    except Exception:
        return part


def add_vertical_fillet(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    radius = sample_uniform(rng, 0.8, min(spec.height * 0.3, 6.5))
    try:
        return part.edges("|Z").fillet(radius)
    except Exception:
        return part


def add_random_features(part: cq.Workplane, spec: StockSpec, rng: np.random.Generator) -> cq.Workplane:
    pocket_ops = rng.integers(1, 4)
    for _ in range(pocket_ops):
        if rng.random() < 0.5:
            part = add_rect_pocket(part, spec, rng)
        else:
            part = add_circular_pocket(part, spec, rng)

    # Optionally add bosses to create steep features.
    if rng.random() < 0.7:
        boss_count = rng.integers(1, 3)
        for _ in range(boss_count):
            part = add_boss(part, spec, rng)

    if rng.random() < 0.6:
        part = add_side_chamfer(part, spec, rng)

    if rng.random() < 0.5:
        part = add_vertical_fillet(part, spec, rng)

    return part


def generate_part(rng: np.random.Generator) -> Tuple[cq.Workplane, StockSpec]:
    part, spec = build_random_stock(rng)
    part = add_random_features(part, spec, rng)
    return part, spec


def export_part(part: cq.Workplane, destination: Path) -> None:
    ensure_dir(destination.parent)
    exporters.export(part, destination, exportType="STL", **STL_EXPORT_ARGS)


def choose_label(mesh_bbox: list[float], rng: np.random.Generator) -> dict:
    xy_extent = max(mesh_bbox[0], mesh_bbox[1], 1e-3)
    z_extent = mesh_bbox[2]
    slenderness = z_extent / xy_extent

    if slenderness < 0.28:
        strategy = "raster"
        angle_deg = sample_uniform(rng, 0.0, 180.0)
        step_over = sample_uniform(rng, TOOL_DIAMETER_MM * 0.35, TOOL_DIAMETER_MM * 0.65)
    else:
        strategy = "waterline"
        angle_deg = 0.0
        step_over = sample_uniform(rng, TOOL_DIAMETER_MM * 0.15, TOOL_DIAMETER_MM * 0.35)

    return {
        "strategy": strategy,
        "angle_deg": round(angle_deg, 2),
        "step_over_mm": round(float(step_over), 3),
    }


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


def build_meta_record(bbox: list[float], label: dict) -> dict:
    return {
        "bbox": [round(v, 3) for v in bbox],
        "material": "MDF",
        "tool_diameter_mm": TOOL_DIAMETER_MM,
        "label": label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic STL + metadata samples.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the dataset.")
    parser.add_argument("--n", type=int, default=300, help="Number of samples to create (default: 300).")
    parser.add_argument("--seed", type=int, default=2025, help="Base RNG seed for reproducibility.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sample folders.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out)

    seed_iter = rolling_seed_sequence(args.seed, args.n)

    for idx in range(args.n):
        sample_seed = next(seed_iter)
        rng = np.random.default_rng(sample_seed)
        sample_dir = prepare_sample_dir(out_dir, idx + 1, force=args.force)

        part, spec = generate_part(rng)
        mesh_path = sample_dir / "mesh.stl"
        export_part(part, mesh_path)

        mesh = load_mesh(mesh_path)
        bbox = compute_bbox(mesh)
        label = choose_label(bbox, rng)

        meta = build_meta_record(bbox, label)
        write_meta(sample_dir / "meta.json", meta)

        print(f"[{idx + 1:04d}/{args.n:04d}] {sample_dir.name} -> {label['strategy']} (bbox={bbox})")

    print(f"Dataset completed: {args.n} samples in {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
