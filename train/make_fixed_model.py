"""
Emit a minimal TorchScript module that always returns a fixed machining decision.

The scripted module is convenient for integration testing because it removes
all sources of randomness and heavy dependencies. Regardless of the input
features it chooses the Raster strategy, reports a 45 degree angle, and scales
the step-over to 40% of the provided tool diameter channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class FixedStrategyModule(nn.Module):
    """
    Deterministic strategy predictor used for pipeline smoke tests.

    The forward method supports both 1D tensors shaped `(6,)` and 2D tensors
    shaped `(batch, 6)`. The implementation only consults the tool diameter
    (channel index 5) to derive the step-over magnitude.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if features.size(-1) < 6:
            raise ValueError("Expected at least 6 feature channels.")

        batch = features.shape[0]
        device = features.device
        dtype = features.dtype

        logits = torch.tensor([8.0, -8.0], dtype=dtype, device=device).expand(batch, 2)
        angle = torch.full((batch, 1), 45.0, dtype=dtype, device=device)
        tool_diameter = features[:, 5:6]
        step_over = tool_diameter * 0.4
        return torch.cat([logits, angle, step_over], dim=1)


def build_module() -> FixedStrategyModule:
    """Instantiate the scripted module."""
    return FixedStrategyModule()


def export(path: Path) -> None:
    """Save the scripted module to the requested path."""
    module = build_module()
    scripted = torch.jit.script(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))
    print(f"Fixed TorchScript model written to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed decision TorchScript model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/fixed_test.pt"),
        help="Target TorchScript file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.output)


if __name__ == "__main__":
    main()

