#!/usr/bin/env python3
"""
Offline evaluation helper for machining strategy models.

The script mirrors the C++ feature extraction path so results can be compared
against the desktop app without requiring a GUI build. Given a directory of
local mesh samples and per-part metadata, it will:

* extract geometric features from each mesh
* run a TorchScript or ONNX model
* compute accuracy, macro-F1, and a confusion matrix
* estimate a naive time proxy (sum(path_length_mm / feedrate_mm_per_min))
* write per-sample results to CSV together with a Markdown summary report

Example:
    python tools/eval/run_eval.py --dataset testdata/eval/smoke \\
        --model models/test_run/strategy_v0.onnx --output build/eval/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import trimesh

try:
    import onnxruntime as ort  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


SLOPE_BOUNDS_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 90.1], dtype=np.float64)
CLASS_NAMES = ["raster", "waterline"]
EPSILON = 1e-6
DEFAULT_FEATURE_ORDER = [
    "bbox_x_mm",
    "bbox_y_mm",
    "bbox_z_mm",
    "surface_area_mm2",
    "volume_mm3",
    "slope_bin_0_15",
    "slope_bin_15_30",
    "slope_bin_30_45",
    "slope_bin_45_60",
    "slope_bin_60_90",
    "mean_curvature_rad",
    "curvature_variance_rad2",
    "flat_area_ratio",
    "steep_area_ratio",
    "pocket_depth_mm",
    "user_step_over_mm",
    "tool_diameter_mm",
]


@dataclass
class ModelCard:
    feature_names: list[str]
    feature_count: int
    normalize_mean: list[float]
    normalize_std: list[float]


@dataclass
class SampleMeta:
    identifier: str
    mesh_path: Path
    strategy: str
    raster_angle_deg: float
    step_over_mm: float
    user_step_over_mm: float
    tool_diameter_mm: float
    path_length_mm: float
    feedrate_mm_per_min: float


@dataclass
class Prediction:
    logits: Optional[np.ndarray]
    angle_deg: Optional[float]
    step_over_mm: Optional[float]
    strategy_index: int


def load_model_card(model_path: Path) -> Optional[ModelCard]:
    """Attempt to load the sidecar model card matching model_path."""
    candidates = [
        model_path.with_suffix(model_path.suffix + ".model.json"),
        model_path.with_suffix(".model.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model card {candidate} is not valid JSON: {exc}") from exc

            features = data.get("features") or {}
            feature_count = int(features.get("count") or 0)
            names = [str(x) for x in features.get("names") or []]
            normalize = features.get("normalize") or {}
            mean = [float(x) for x in normalize.get("mean") or []]
            std = [float(x) for x in normalize.get("std") or []]

            if not names:
                names = list(DEFAULT_FEATURE_ORDER)
            if feature_count <= 0:
                feature_count = len(names)
            if len(names) != feature_count:
                raise ValueError(
                    f"Model card {candidate} lists {len(names)} names but count={feature_count}."
                )
            if mean and len(mean) != feature_count:
                raise ValueError(
                    f"Model card {candidate} normalize.mean has {len(mean)} entries, expected {feature_count}."
                )
            if std and len(std) != feature_count:
                raise ValueError(
                    f"Model card {candidate} normalize.std has {len(std)} entries, expected {feature_count}."
                )

            return ModelCard(names, feature_count, mean, std)
    return None


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh using trimesh, collapsing scenes to a single mesh."""
    mesh = trimesh.load_mesh(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangle mesh.")
    mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise ValueError(f"{path} has no geometry to evaluate.")
    mesh.rezero()
    return mesh


def compute_vertex_normals(faces: np.ndarray, face_normals: np.ndarray, vertex_count: int) -> np.ndarray:
    """Reconstruct per-vertex normals by averaging adjacent face normals."""
    vertex_normals = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    mask = lengths > EPSILON
    vertex_normals[mask] /= lengths[mask, None]
    vertex_normals[~mask] = 0.0
    return vertex_normals


def compute_global_features(mesh: trimesh.Trimesh) -> dict[str, float]:
    """
    Replicate ai::FeatureExtractor::computeGlobalFeatures for compatibility with the app.
    """
    vertices = mesh.vertices.astype(np.float64)
    faces = mesh.faces.astype(np.int64)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh is empty.")

    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    edge1 = p1 - p0
    edge2 = p2 - p0
    cross = np.cross(edge1, edge2)
    cross_norm = np.linalg.norm(cross, axis=1)

    valid_face_mask = cross_norm > EPSILON
    if not np.any(valid_face_mask):
        raise ValueError("Mesh has zero-area faces only.")

    tri_area = 0.5 * cross_norm
    surface_area = float(tri_area.sum())
    if surface_area <= EPSILON:
        raise ValueError("Surface area is zero after filtering degenerate triangles.")

    face_normals = np.zeros_like(cross)
    face_normals[valid_face_mask] = cross[valid_face_mask] / cross_norm[valid_face_mask][:, None]

    vertex_normals = compute_vertex_normals(faces, face_normals, vertices.shape[0])

    bbox_extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    slopes_rad = np.arccos(
        np.clip(np.abs(face_normals[valid_face_mask, 2]), 0.0, 1.0)
    )
    slopes_deg = np.degrees(slopes_rad)
    slope_bins = np.digitize(slopes_deg, SLOPE_BOUNDS_DEG[1:-1], right=False)

    tri_area_valid = tri_area[valid_face_mask]
    slope_area = np.zeros(5, dtype=np.float64)
    np.add.at(slope_area, slope_bins, tri_area_valid)
    slope_histogram = slope_area / surface_area

    flat_mask = slopes_deg < 15.0
    flat_area = float(tri_area_valid[flat_mask].sum()) if np.any(flat_mask) else 0.0
    steep_mask = slopes_deg >= 60.0
    steep_area = float(tri_area_valid[steep_mask].sum()) if np.any(steep_mask) else 0.0

    cross_p1_p2 = np.cross(p1, p2)
    enclosed_volume = float(np.einsum("ij,ij->i", p0, cross_p1_p2).sum() / 6.0)

    faces_valid = faces[valid_face_mask]
    face_normals_valid = face_normals[valid_face_mask]
    vertex_normals_valid = vertex_normals[faces_valid]
    vertex_valid_mask = np.linalg.norm(vertex_normals_valid, axis=2) > EPSILON
    dots = np.einsum("fij,fj->fi", vertex_normals_valid, face_normals_valid)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    curvature_samples = angles[vertex_valid_mask]

    if curvature_samples.size:
        mean_curvature = float(curvature_samples.mean())
        curvature_variance = float(curvature_samples.var())
    else:
        mean_curvature = 0.0
        curvature_variance = 0.0

    return {
        "bbox_x_mm": float(bbox_extent[0]),
        "bbox_y_mm": float(bbox_extent[1]),
        "bbox_z_mm": float(bbox_extent[2]),
        "surface_area_mm2": float(surface_area),
        "volume_mm3": float(abs(enclosed_volume)),
        "slope_bin_0_15": float(slope_histogram[0]),
        "slope_bin_15_30": float(slope_histogram[1]),
        "slope_bin_30_45": float(slope_histogram[2]),
        "slope_bin_45_60": float(slope_histogram[3]),
        "slope_bin_60_90": float(slope_histogram[4]),
        "mean_curvature_rad": mean_curvature,
        "curvature_variance_rad2": curvature_variance,
        "flat_area_ratio": float(flat_area / surface_area),
        "steep_area_ratio": float(steep_area / surface_area),
        "pocket_depth_mm": float(max(max_z - min_z, 0.0)),
    }


def build_feature_vector(
    base_features: dict[str, float],
    meta: SampleMeta,
    feature_order: list[str],
    expected_count: int,
) -> np.ndarray:
    """Assemble the feature vector respecting the model's expected order."""
    features = dict(base_features)
    features["user_step_over_mm"] = float(meta.user_step_over_mm)
    features["tool_diameter_mm"] = float(meta.tool_diameter_mm)

    vector = [float(features.get(name, 0.0)) for name in feature_order]
    if len(vector) < expected_count:
        vector.extend([0.0] * (expected_count - len(vector)))
    elif len(vector) > expected_count:
        vector = vector[:expected_count]
    return np.asarray(vector, dtype=np.float32)


def infer_onnx(session: "ort.InferenceSession", features: np.ndarray) -> Prediction:
    """Run inference through an ONNX session."""
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model exposes no inputs.")
    input_name = inputs[0].name

    output_info = session.get_outputs()
    output_names = [output.name for output in output_info]
    results = session.run(output_names or None, {input_name: features[None, :]})

    result_map = {name: value for name, value in zip(output_names, results)}

    logits = result_map.get("logits")
    angle = result_map.get("angle_deg")
    step = result_map.get("step_over_mm")
    logits_vec = None
    if logits is not None:
        logits_vec = np.asarray(logits, dtype=np.float32).reshape(-1)
    angle_val = float(angle.reshape(-1)[0]) if angle is not None else None  # type: ignore[attr-defined]
    step_val = float(step.reshape(-1)[0]) if step is not None else None  # type: ignore[attr-defined]

    strategy_index = 0
    if logits_vec is not None and logits_vec.size >= 2:
        strategy_index = int(np.argmax(logits_vec))

    return Prediction(logits_vec, angle_val, step_val, strategy_index)


def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    return tensor.detach().cpu().numpy()


def infer_torch(
    module: "torch.jit.ScriptModule",
    features: np.ndarray,
    device: "torch.device",
) -> Prediction:
    if torch is None:
        raise RuntimeError("PyTorch is not available but a TorchScript model was provided.")

    module.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(features[None, :]).to(device)
        output = module(input_tensor)

    logits_tensor = None
    angle_tensor = None
    step_tensor = None

    if isinstance(output, dict):
        logits_tensor = output.get("logits")
        angle_tensor = output.get("angle")
        if angle_tensor is None:
            angle_tensor = output.get("angle_deg")
        step_tensor = output.get("stepover") or output.get("step_over_mm")
    elif isinstance(output, (list, tuple)):
        if len(output) >= 1:
            logits_tensor = output[0]
        if len(output) >= 2:
            angle_tensor = output[1]
        if len(output) >= 3:
            step_tensor = output[2]
    elif torch.is_tensor(output):
        logits_tensor = output

    logits_vec = None
    if logits_tensor is not None:
        logits_vec = _to_numpy(logits_tensor).reshape(-1)
    angle_val = float(_to_numpy(angle_tensor).reshape(-1)[0]) if angle_tensor is not None else None
    step_val = float(_to_numpy(step_tensor).reshape(-1)[0]) if step_tensor is not None else None

    strategy_index = 0
    if logits_vec is not None and logits_vec.size >= 2:
        strategy_index = int(np.argmax(logits_vec))

    return Prediction(logits_vec, angle_val, step_val, strategy_index)


def load_sample_meta(path: Path) -> SampleMeta:
    data = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(data.get("id") or path.stem)
    mesh_name = data.get("mesh") or f"{path.stem}.stl"
    mesh_path = (path.parent / mesh_name).resolve()

    def read_float(key: str, default: float = 0.0) -> float:
        value = data.get(key, default)
        if value is None:
            return float(default)
        return float(value)

    strategy = str(data.get("strategy") or "").strip().lower()
    if strategy not in CLASS_NAMES:
        raise ValueError(f"{path}: strategy must be one of {CLASS_NAMES}, got {strategy!r}")

    return SampleMeta(
        identifier=identifier,
        mesh_path=mesh_path,
        strategy=strategy,
        raster_angle_deg=read_float("raster_angle_deg"),
        step_over_mm=read_float("step_over_mm"),
        user_step_over_mm=read_float("user_step_over_mm", read_float("step_over_mm")),
        tool_diameter_mm=read_float("tool_diameter_mm"),
        path_length_mm=read_float("path_length_mm"),
        feedrate_mm_per_min=read_float("feedrate_mm_per_min"),
    )


def iter_samples(dataset_dir: Path) -> Iterable[SampleMeta]:
    for meta_path in sorted(dataset_dir.glob("*.json")):
        yield load_sample_meta(meta_path)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def compute_confusion_matrix(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def compute_macro_f1(confusion: np.ndarray) -> float:
    f_scores: list[float] = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        if tp == 0.0 and (fp == 0.0 or fn == 0.0):
            f_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        if precision + recall == 0.0:
            f_scores.append(0.0)
        else:
            f_scores.append(2.0 * precision * recall / (precision + recall))
    if not f_scores:
        return 0.0
    return float(statistics.mean(f_scores))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_confusion_table(confusion: np.ndarray) -> str:
    header = "| truth/pred | " + " | ".join(CLASS_NAMES) + " |\n"
    divider = "|" + " --- |" * (len(CLASS_NAMES) + 1) + "\n"
    body_lines = []
    for idx, name in enumerate(CLASS_NAMES):
        cells = " | ".join(str(int(v)) for v in confusion[idx])
        body_lines.append(f"| {name} | {cells} |")
    return header + divider + "\n".join(body_lines)


def write_markdown_report(
    path: Path,
    *,
    sample_count: int,
    accuracy: float,
    macro_f1: float,
    confusion: np.ndarray,
    time_proxy_minutes: float,
    avg_angle_error: float,
    avg_step_error: float,
) -> None:
    report = [
        "# Offline Strategy Evaluation",
        "",
        f"*Samples evaluated*: {sample_count}",
        f"*Accuracy*: {accuracy:.3f}",
        f"*Macro-F1*: {macro_f1:.3f}",
        f"*Total time proxy* (minutes): {time_proxy_minutes:.2f}",
        f"*Mean |angle error|* (deg): {avg_angle_error:.2f}",
        f"*Mean |step-over error|* (mm): {avg_step_error:.3f}",
        "",
        "## Confusion Matrix",
        "",
        format_confusion_table(confusion),
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for machining strategy models.")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing *.json metadata + meshes.")
    parser.add_argument("--model", type=Path, required=True, help="Path to TorchScript (.pt) or ONNX (.onnx) model.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for CSV + Markdown outputs.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Preferred inference device. 'auto' uses CUDA when available.",
    )

    args = parser.parse_args()
    dataset_dir: Path = args.dataset.resolve()
    model_path: Path = args.model.resolve()
    output_dir: Path = args.output.resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory {dataset_dir} does not exist.")
    if not model_path.is_file():
        raise SystemExit(f"Model file {model_path} does not exist.")

    card = load_model_card(model_path)
    feature_names = card.feature_names if card else list(DEFAULT_FEATURE_ORDER)
    feature_count = card.feature_count if card else len(feature_names)

    ensure_output_dir(output_dir)

    backend = model_path.suffix.lower()
    onnx_session = None
    torch_module = None
    torch_device: Optional["torch.device"] = None
    if backend == ".onnx":
        if ort is None:
            raise SystemExit("onnxruntime is not installed but an ONNX model was supplied.")
        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime providers available: {available_providers}")
        if args.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif args.device == "cuda":
            if "CUDAExecutionProvider" not in available_providers:
                raise SystemExit("CUDA requested but CUDAExecutionProvider is not available in this build.")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )
        print(f"ONNX Runtime providers selected: {providers}")
        onnx_session = ort.InferenceSession(model_path.as_posix(), providers=providers)
    elif backend == ".pt":
        if torch is None:
            raise SystemExit("PyTorch is not installed but a TorchScript model was supplied.")
        cuda_available = torch.cuda.is_available()
        if args.device == "cuda":
            if not cuda_available:
                raise SystemExit("CUDA requested but torch.cuda.is_available() returned False.")
            torch_device = torch.device("cuda")
        elif args.device == "auto" and cuda_available:
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")
        print(f"Torch CUDA available: {cuda_available}, using device: {torch_device}")
        torch_module = torch.jit.load(model_path.as_posix(), map_location=torch_device)
        torch_module.to(torch_device)
    else:
        raise SystemExit(f"Unsupported model extension {backend}. Use .onnx or .pt.")

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    angle_errors: list[float] = []
    step_errors: list[float] = []
    total_time_proxy = 0.0

    for sample in iter_samples(dataset_dir):
        mesh = load_mesh(sample.mesh_path)
        base_feats = compute_global_features(mesh)
        feature_vec = build_feature_vector(base_feats, sample, feature_names, feature_count)

        if onnx_session is not None:
            prediction = infer_onnx(onnx_session, feature_vec)
        else:
            assert torch_module is not None and torch_device is not None
            prediction = infer_torch(torch_module, feature_vec, torch_device)  # type: ignore[arg-type]

        true_index = CLASS_NAMES.index(sample.strategy)
        pred_index = prediction.strategy_index

        predicted_strategy = CLASS_NAMES[pred_index]
        logits = prediction.logits.tolist() if prediction.logits is not None else []

        time_proxy = 0.0
        if sample.feedrate_mm_per_min > 0.0:
            time_proxy = sample.path_length_mm / sample.feedrate_mm_per_min
        total_time_proxy += time_proxy

        angle_error = None
        if prediction.angle_deg is not None:
            angle_error = abs(prediction.angle_deg - sample.raster_angle_deg)
            angle_errors.append(angle_error)
        step_error = None
        if prediction.step_over_mm is not None:
            step_error = abs(prediction.step_over_mm - sample.step_over_mm)
            step_errors.append(step_error)

        rows.append(
            {
                "id": sample.identifier,
                "strategy_true": sample.strategy,
                "strategy_pred": predicted_strategy,
                "is_correct": bool(true_index == pred_index),
                "logits": logits,
                "angle_true_deg": sample.raster_angle_deg,
                "angle_pred_deg": prediction.angle_deg,
                "step_true_mm": sample.step_over_mm,
                "step_pred_mm": prediction.step_over_mm,
                "angle_abs_error_deg": angle_error,
                "step_abs_error_mm": step_error,
                "path_length_mm": sample.path_length_mm,
                "feedrate_mm_per_min": sample.feedrate_mm_per_min,
                "time_proxy_min": time_proxy,
                "user_step_over_mm": sample.user_step_over_mm,
                "tool_diameter_mm": sample.tool_diameter_mm,
            }
        )

        y_true.append(true_index)
        y_pred.append(pred_index)

    sample_count = len(rows)
    if sample_count == 0:
        raise SystemExit(f"No samples found under {dataset_dir}.")

    accuracy = float(sum(1 for row in rows if row["is_correct"]) / sample_count)
    confusion = compute_confusion_matrix(y_true, y_pred)
    macro_f1 = compute_macro_f1(confusion)

    avg_angle_error = float(statistics.mean(angle_errors)) if angle_errors else 0.0
    avg_step_error = float(statistics.mean(step_errors)) if step_errors else 0.0

    write_csv(rows, output_dir / "metrics.csv")
    write_markdown_report(
        output_dir / "report.md",
        sample_count=sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion=confusion,
        time_proxy_minutes=total_time_proxy,
        avg_angle_error=avg_angle_error,
        avg_step_error=avg_step_error,
    )

    print(f"Evaluated {sample_count} samples.")
    print(f"Accuracy: {accuracy:.3f}, Macro-F1: {macro_f1:.3f}")
    print(f"Total time proxy (minutes): {total_time_proxy:.2f}")
    print(f"Report written to {output_dir}")


if __name__ == "__main__":
    main()
