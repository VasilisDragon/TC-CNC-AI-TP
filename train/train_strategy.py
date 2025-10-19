"""
Train a small MLP that predicts machining strategy logits, raster angle, and
step-over distance from simple geometric and tooling features.

The script generates a synthetic dataset on the fly using heuristics that mimic
the dataset generator in this repository, trains a multi-head model, and
exports both TorchScript and ONNX artefacts alongside lightweight metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import onnx


# Ensure UTF-8 stdout to avoid ONNX export Unicode issues on Windows consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        # Some IDEs / redirected stdout may not support reconfigure.
        pass

FEATURE_NAMES_V1: list[str] = [
    "bbox_x_mm",
    "bbox_y_mm",
    "bbox_z_mm",
    "surface_area_mm2",
    "user_step_over_mm",
    "tool_diameter_mm",
]

FEATURE_NAMES_V2: list[str] = [
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

EXPECTED_FEATURE_COUNT = 17


def get_feature_names(use_v2: bool) -> list[str]:
    return list(FEATURE_NAMES_V2 if use_v2 else FEATURE_NAMES_V1)


def build_feature_scale(use_v2: bool) -> torch.Tensor:
    if use_v2:
        values = [
            100.0,  # bbox_x_mm
            100.0,  # bbox_y_mm
            100.0,  # bbox_z_mm
            10000.0,  # surface area
            1_000_000.0,  # volume
            1.0,  # slope bin 0-15
            1.0,  # slope bin 15-30
            1.0,  # slope bin 30-45
            1.0,  # slope bin 45-60
            1.0,  # slope bin 60-90
            1.0,  # mean curvature
            1.0,  # curvature variance
            1.0,  # flat ratio
            1.0,  # steep ratio
            100.0,  # pocket depth
            10.0,  # user step-over
            10.0,  # tool diameter
        ]
    else:
        values = [100.0, 100.0, 100.0, 10000.0, 10.0, 10.0]
    return torch.tensor(values, dtype=torch.float32)


def set_seed(seed: int) -> None:
    """Configure RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not support this flag.
        pass
    torch.backends.cudnn.benchmark = False


def generate_sample(rng: np.random.Generator, use_v2: bool) -> Tuple[np.ndarray, int, float, float]:
    """
    Produce a single synthetic training sample.

    Returns:
        features: float32 vector of length 6 (v1) or 17 (v2)
        strategy_idx: 0 (raster) or 1 (waterline)
        angle_deg: target finishing angle in degrees
        step_over_mm: desired step-over in millimetres
    """
    # Sample bounding box dimensions (mm) and ensure x >= y ordering for stability.
    bbox_x = float(rng.uniform(60.0, 140.0))
    bbox_y = float(rng.uniform(40.0, 110.0))
    bbox_z = float(rng.uniform(15.0, 60.0))
    if bbox_y > bbox_x:
        bbox_x, bbox_y = bbox_y, bbox_x

    surface_area = 2.0 * (bbox_x * bbox_y + bbox_x * bbox_z + bbox_y * bbox_z)
    volume = bbox_x * bbox_y * bbox_z

    tool_diameter = float(rng.uniform(3.0, 12.0))
    step_hint = float(tool_diameter * rng.uniform(0.35, 0.55))

    flatness_ratio = bbox_z / max(bbox_x, bbox_y)
    aspect_zx = bbox_z / max(bbox_x, 1e-6)

    if use_v2:
        slope_bins = rng.dirichlet(np.ones(5, dtype=np.float32)).astype(np.float32)
        flat_ratio = float(slope_bins[0])
        steep_ratio = float(slope_bins[-1])
        mean_curv = float(abs(rng.normal(loc=0.08, scale=0.04)))
        var_curv = float(abs(rng.normal(loc=0.02, scale=0.01)))
        pocket_depth = float(bbox_z * rng.uniform(0.6, 1.0))
        area_with_noise = float(surface_area * rng.uniform(0.85, 1.15))
        volume_with_noise = float(volume * rng.uniform(0.85, 1.15))
        head = np.array(
            [bbox_x, bbox_y, bbox_z, area_with_noise, volume_with_noise],
            dtype=np.float32,
        )
        tail = np.array(
            [
                mean_curv,
                var_curv,
                flat_ratio,
                steep_ratio,
                pocket_depth,
                step_hint,
                tool_diameter,
            ],
            dtype=np.float32,
        )
        features = np.concatenate((head, slope_bins, tail))
    else:
        features = np.array(
            [
                bbox_x,
                bbox_y,
                bbox_z,
                surface_area,
                step_hint,
                tool_diameter,
            ],
            dtype=np.float32,
        )

    # Label heuristics influenced by flatness and aspect metrics.
    # Introduce mild stochasticity so training is non-trivial.
    steepness = float(rng.uniform(0.0, 1.0))
    combined_indicator = 0.5 * flatness_ratio + 0.3 * aspect_zx + 0.2 * steepness
    decision = combined_indicator + float(rng.normal(0.0, 0.05))

    base_angle = float(np.degrees(np.arctan2(bbox_y, bbox_x)))

    if decision < 0.45:
        # Raster strategy: prefer larger step over with varied angle.
        strategy_idx = 0
        angle_deg = float(np.clip(base_angle + rng.normal(0.0, 8.0), 0.0, 180.0))
        step_over = min(
            float(step_hint * rng.uniform(0.9, 1.1)),
            float(tool_diameter * 0.9),
        )
    else:
        # Waterline strategy: neutral angle, conservative step over.
        strategy_idx = 1
        angle_deg = float(np.clip(rng.uniform(0.0, 5.0) + 15.0 * flatness_ratio, 0.0, 45.0))
        step_over = min(
            float(step_hint * rng.uniform(0.35, 0.65)),
            float(tool_diameter * 0.7),
        )

    # Enforce positive step-over and keep angle within the documented 0-180 range.
    step_over = float(max(step_over, tool_diameter * 0.1))
    angle_deg = float(np.clip(angle_deg, 0.0, 180.0))

    return features.astype(np.float32, copy=False), strategy_idx, angle_deg, step_over


def build_dataset(num_samples: int, seed: int, use_v2: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate an entire synthetic dataset."""
    rng = np.random.default_rng(seed)
    features = []
    strategies = []
    angles = []
    steps = []
    for _ in range(num_samples):
        feat, strategy, angle, step = generate_sample(rng, use_v2)
        features.append(feat)
        strategies.append(strategy)
        angles.append(angle)
        steps.append(step)
    return (
        np.stack(features, axis=0),
        np.array(strategies, dtype=np.int64),
        np.array(angles, dtype=np.float32),
        np.array(steps, dtype=np.float32),
    )


class StrategyDataset(Dataset):
    """PyTorch dataset wrapper for the synthetic samples."""

    def __init__(
        self,
        features: np.ndarray,
        strategies: np.ndarray,
        angles: np.ndarray,
        steps: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features).float()
        self.strategies = torch.from_numpy(strategies).long()
        self.angles = torch.from_numpy(angles).float()
        self.steps = torch.from_numpy(steps).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.strategies[idx],
            self.angles[idx],
            self.steps[idx],
        )


class StrategyModel(nn.Module):
    """Two-layer MLP with task-specific heads."""

    def __init__(self, input_dim: int, tool_index: int, feature_scale: torch.Tensor) -> None:
        super().__init__()
        if feature_scale.numel() != input_dim:
            raise ValueError(
                f"StrategyModel expected feature_scale with {input_dim} elements, got {feature_scale.numel()}"
            )
        if not 0 <= tool_index < input_dim:
            raise ValueError(f"Tool diameter index {tool_index} is out of range for {input_dim} features.")

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.head_cls = nn.Linear(64, 2)
        self.head_angle = nn.Linear(64, 1)
        self.head_step = nn.Linear(64, 1)
        self.register_buffer("feature_scale", feature_scale.clone().detach().float())
        self.tool_index = int(tool_index)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.feature_scale.to(features.dtype)
        latent = self.backbone(features / scale)
        logits = self.head_cls(latent)
        angle = torch.sigmoid(self.head_angle(latent)) * 180.0
        tool_diameter = features[:, self.tool_index : self.tool_index + 1]
        step_over = torch.sigmoid(self.head_step(latent)) * (tool_diameter * 0.9)
        return logits, angle, step_over


@dataclass
class Metrics:
    loss: float
    cls_loss: float
    angle_mae: float
    step_mae: float
    accuracy: float


def evaluate(
    model: StrategyModel,
    loader: DataLoader,
    device: torch.device,
    criterion_cls: nn.Module,
    l1_loss: nn.Module,
    angle_weight: float,
    step_weight: float,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_angle = 0.0
    total_step = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, strategy, angle, step in loader:
            features = features.to(device)
            strategy = strategy.to(device)
            angle = angle.to(device)
            step = step.to(device)

            logits, pred_angle, pred_step = model(features)
            cls_loss = criterion_cls(logits, strategy)
            angle_loss = l1_loss(pred_angle.squeeze(1), angle)
            step_loss = l1_loss(pred_step.squeeze(1), step)
            loss = cls_loss + angle_weight * angle_loss + step_weight * step_loss

            total_loss += loss.item() * features.size(0)
            total_cls += cls_loss.item() * features.size(0)
            total_angle += angle_loss.item() * features.size(0)
            total_step += step_loss.item() * features.size(0)

            predicted = logits.argmax(dim=1)
            correct += (predicted == strategy).sum().item()
            total += strategy.size(0)

    if total == 0:
        return Metrics(loss=0.0, cls_loss=0.0, angle_mae=0.0, step_mae=0.0, accuracy=0.0)

    return Metrics(
        loss=total_loss / total,
        cls_loss=total_cls / total,
        angle_mae=total_angle / total,
        step_mae=total_step / total,
        accuracy=correct / total,
    )


def train_model(
    dataset: StrategyDataset,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    angle_weight: float,
    step_weight: float,
    lr: float,
    val_split: float,
    seed: int,
    tool_index: int,
    feature_scale: torch.Tensor,
) -> Tuple[StrategyModel, Metrics, int]:
    """Train the model with early stopping."""
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples.")

    val_size = max(1, int(dataset_size * val_split))
    if val_size >= dataset_size:
        val_size = dataset_size - 1
    train_size = dataset_size - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = StrategyModel(dataset.features.shape[1], tool_index, feature_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for features, strategy, angle, step in train_loader:
            features = features.to(device)
            strategy = strategy.to(device)
            angle = angle.to(device)
            step = step.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, pred_angle, pred_step = model(features)
            cls_loss = criterion_cls(logits, strategy)
            angle_loss = l1_loss(pred_angle.squeeze(1), angle)
            step_loss = l1_loss(pred_step.squeeze(1), step)
            loss = cls_loss + angle_weight * angle_loss + step_weight * step_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            total += features.size(0)

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            criterion_cls,
            l1_loss,
            angle_weight,
            step_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion_cls,
            l1_loss,
            angle_weight,
            step_weight,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.3f}"
        )

        if val_metrics.loss + 1e-6 < best_val_loss:
            best_val_loss = val_metrics.loss
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_state is None:
        best_state = {"model": model.state_dict(), "epoch": epochs, "metrics": val_metrics}

    model.load_state_dict(best_state["model"])
    return model, best_state["metrics"], best_state["epoch"]


def export_torchscript(model: StrategyModel, example: torch.Tensor, path: Path) -> None:
    """Export the model to TorchScript via tracing."""
    model.eval()
    traced = torch.jit.trace(model, example)
    traced.save(str(path))


def export_onnx(model: StrategyModel, example: torch.Tensor, path: Path, opset: int) -> None:
    """Export the model to ONNX with dynamic batch support."""
    model.eval()
    torch.onnx.export(
        model,
        example,
        str(path),
        opset_version=opset,
        input_names=["features"],
        output_names=["logits", "angle_deg", "step_over_mm"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
            "angle_deg": {0: "batch"},
            "step_over_mm": {0: "batch"},
        },
    )
    onnx_model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(onnx_model, str(path), save_as_external_data=False)
    external_data_path = Path(str(path) + ".data")
    if external_data_path.exists():
        external_data_path.unlink()


def write_json(path: Path, payload: dict) -> None:
    """Persist JSON with deterministic formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_interface_schema(feature_names: list[str]) -> Dict[str, object]:
    """Describe the expected ONNX/TorchScript interface."""
    input_dim = len(feature_names)
    description = "Feature vector order: " + ", ".join(feature_names)
    schema: Dict[str, object] = {
        "inputs": [
            {
                "name": "features",
                "shape": ["batch", input_dim],
                "dtype": "float32",
                "description": description,
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "shape": ["batch", 2],
                "dtype": "float32",
                "description": "Strategy logits ordered as [raster, waterline].",
            },
            {
                "name": "angle_deg",
                "shape": ["batch", 1],
                "dtype": "float32",
                "description": "Predicted raster hatch angle in degrees.",
            },
            {
                "name": "step_over_mm",
                "shape": ["batch", 1],
                "dtype": "float32",
                "description": "Predicted step-over distance in millimetres.",
            },
        ],
    }
    schema["feature_names"] = feature_names
    return schema


def make_model_card(
    model_type: str,
    framework: str,
    versions: list[str],
    feature_names: list[str],
    feature_scale: torch.Tensor,
    dataset_id: str,
    dataset_sha: str,
    created_at: str,
    metrics: Metrics,
) -> dict:
    """Construct a schema-compliant model card for the trained artefact."""

    feature_count = len(feature_names)
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"model card schema expects {EXPECTED_FEATURE_COUNT} features, got {feature_count}."
        )
    if feature_scale.numel() != feature_count:
        raise ValueError(
            f"feature_scale must contain {feature_count} entries, got {feature_scale.numel()}."
        )

    std_values = [float(value) for value in feature_scale.tolist()]
    return {
        "schema_version": "1.0.0",
        "model_type": model_type,
        "features": {
            "count": feature_count,
            "names": feature_names,
            "normalize": {
                "mean": [0.0] * feature_count,
                "std": std_values,
            },
        },
        "training": {
            "framework": framework,
            "versions": versions,
        },
        "dataset": {
            "id": dataset_id,
            "sha256": dataset_sha,
        },
        "created_at": created_at,
        "validation": {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train machining strategy predictor.")
    parser.add_argument("--seed", type=int, default=1337, help="Base RNG seed.")
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic samples to generate.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Adam learning rate.")
    parser.add_argument("--angle-weight", type=float, default=0.1, help="Weight for angle L1 loss.")
    parser.add_argument("--step-weight", type=float, default=2.0, help="Weight for step-over L1 loss.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument(
        "--v2-features",
        action="store_true",
        help="Use extended geometry-aware features (bbox, slopes, curvature, pocket depth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store trained artefacts.",
    )
    parser.add_argument("--torchscript-name", type=str, default="strategy_v0.pt", help="TorchScript file name.")
    parser.add_argument("--onnx-name", type=str, default="strategy_v0.onnx", help="ONNX file name.")
    parser.add_argument("--onnx-json-name", type=str, default="strategy_v0.onnx.json", help="ONNX schema JSON name.")
    parser.add_argument(
        "--torch-card-name",
        type=str,
        default="strategy_v0.pt.model.json",
        help="TorchScript model card JSON name.",
    )
    parser.add_argument(
        "--onnx-card-name",
        type=str,
        default="strategy_v0.onnx.model.json",
        help="ONNX model card JSON name.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu or cuda).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    use_v2 = args.v2_features
    feature_names = get_feature_names(use_v2)
    feature_scale = build_feature_scale(use_v2)
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Model card schema requires the v2 feature set (17 features). Rerun with --v2-features."
        )
    tool_index = len(feature_names) - 1
    feature_version = "v2" if use_v2 else "v1"

    set_seed(args.seed)
    features, strategies, angles, steps = build_dataset(args.samples, args.seed, use_v2)
    dataset = StrategyDataset(features, strategies, angles, steps)

    if features.shape[0] > 0:
        preview = ", ".join(f"{v:.3f}" for v in features[0][: min(5, features.shape[1])])
        print(f"[train_strategy] Feature dim: {features.shape[1]} ({feature_version}); preview: [{preview}]")

    model, val_metrics, _best_epoch = train_model(
        dataset,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        angle_weight=args.angle_weight,
        step_weight=args.step_weight,
        lr=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        tool_index=tool_index,
        feature_scale=feature_scale,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.from_numpy(features[:1]).float()

    model_cpu = model.to("cpu")
    example_cpu = example.to("cpu")

    torchscript_path = args.output_dir / args.torchscript_name
    export_torchscript(model_cpu, example_cpu, torchscript_path)

    onnx_path = args.output_dir / args.onnx_name
    export_onnx(model_cpu, example_cpu, onnx_path, args.opset)

    schema = build_interface_schema(feature_names)
    write_json(args.output_dir / args.onnx_json_name, schema)

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dataset_path = getattr(args, "dataset_path", "")
    if dataset_path:
        dataset_id = Path(dataset_path).name
    else:
        dataset_id = f"synthetic_{args.samples}"
    dataset_digest = hashlib.sha256(f"{dataset_id}:{args.samples}:{args.seed}".encode("utf-8")).hexdigest()

    torch_card = make_model_card(
        model_type="torchscript",
        framework="PyTorch",
        versions=[torch.__version__],
        feature_names=feature_names,
        feature_scale=feature_scale,
        dataset_id=dataset_id,
        dataset_sha=dataset_digest,
        created_at=created_at,
        metrics=val_metrics,
    )

    try:
        import onnxruntime as ort  # type: ignore

        onnx_versions = [getattr(ort, "__version__", "unknown")]
    except Exception:
        onnx_versions = [f"opset-{args.opset}"]

    onnx_card = make_model_card(
        model_type="onnx",
        framework="ONNXRuntime",
        versions=onnx_versions,
        feature_names=feature_names,
        feature_scale=feature_scale,
        dataset_id=dataset_id,
        dataset_sha=dataset_digest,
        created_at=created_at,
        metrics=val_metrics,
    )

    write_json(args.output_dir / args.torch_card_name, torch_card)
    write_json(args.output_dir / args.onnx_card_name, onnx_card)

    print(f"TorchScript saved to: {torchscript_path}")
    print(f"ONNX saved to      : {onnx_path}")
    print(f"Schema JSON saved to: {args.output_dir / args.onnx_json_name}")
    print(f"Torch card saved to : {args.output_dir / args.torch_card_name}")
    print(f"ONNX card saved to  : {args.output_dir / args.onnx_card_name}")


if __name__ == "__main__":
    main()
