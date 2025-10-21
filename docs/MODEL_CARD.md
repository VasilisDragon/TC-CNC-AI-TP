# TorchAI Model Card

## Overview
TorchAI wraps TorchScript modules so the CNC toolpath planner can predict machining strategy parameters from geometric features. When LibTorch is available the engine loads a scripted `.pt` module at startup, evaluates it on demand, and feeds the predicted strategy back into the placeholder toolpath generator. If LibTorch is absent the code compiles in a stub mode that always returns safe defaults, keeping the public ABI intact.

## Expected Inputs
TorchAI currently assembles a single `1 x 6` float tensor per inference. All values are expressed in millimetres unless noted.

1. Bounding box size X
2. Bounding box size Y
3. Bounding box size Z
4. Approximate surface area (triangle mesh estimate)
5. User selected step-over
6. Active tool diameter

The tensor is ordered exactly as listed above. Future revisions will append additional channels (for example, multi-view renders or point-cloud features) but the leading elements will remain stable.

## Expected Outputs
The scripted module may return one of the following structures:

- A dictionary containing keys `logits`, `angle`, and `stepover`
- A tuple or list with three elements in the same order
- A flat tensor with at least four values, interpreted as `[logit0, logit1, angle, stepover]`

`logits` must contain two values where index `0` corresponds to the Raster strategy and index `1` corresponds to Waterline. TorchAI applies `softmax` to pick the dominant strategy. `angle` and `stepover` are treated as scalar regressions (degrees and millimetres respectively). Any missing or invalid field causes the adapter to fall back to the user supplied defaults.

## Production Model (`models/strategy_v0.pt` / `.onnx`)

- Forward signature: expects a `1 x 6` or `N x 6` float tensor ordered as described in *Expected Inputs*. Internally the module normalises each channel before applying two fully connected layers.
- Outputs:
  1. `logits` &mdash; unnormalised scores for Raster vs Waterline.
  2. `angle_deg` &mdash; scalar angle in `[0, 180]`.
  3. `step_over_mm` &mdash; scalar step-over constrained to `<= 0.9 * tool_diameter_mm`.
- Training data: 2,000 synthetic samples generated with the heuristics in `train/train_strategy.py`. The dataset mirrors the C++ feature order so the exported model can be consumed without additional preprocessing.
- Validation snapshot (seed 1337, opset 17 export):
  * Accuracy: **0.81**
  * Angle MAE: **8.75 deg**
  * Step-over MAE: **0.48 mm**

## Fixed Test Model (`models/fixed_test.pt`)

To exercise the C++ integration without running the full trainer, a scripted module is shipped alongside the repo:

- Forward signature: accepts `1 x 6` or `N x 6` float tensors matching the feature order above.
- Output tensor shape: `N x 4`, ordered as `[logit_raster, logit_waterline, angle_deg, step_over_mm]`.
- Behaviour: always favours the Raster strategy (`logit_raster = +8`, `logit_waterline = -8`), reports a constant `45.0 deg` angle, and derives the step-over as `0.4 * tool_diameter` using channel index `5` from the input tensor.

This deterministic module is useful for smoke tests because it has no learned weights and therefore loads instantly on every platform.

## Device Selection
At construction time TorchAI loads the module on CPU and attempts to move it to CUDA if:

- LibTorch was built with CUDA support
- CUDA devices are visible at runtime
- The user has not enabled **Force CPU inference** in the preferences dialog

The active device label is exposed through `TorchAI::device()` and mirrored in **AI > Preferences...**. Toggling the force-CPU checkbox immediately migrates the module for preview; accepting the dialog persists the setting via `QSettings`.

## Failure Handling
Any load or inference error is caught, logged, and stored in `TorchAI::lastError()`. On failure the adapter returns a conservative two-step plan:

1. Raster roughing at 45 degrees using the user supplied step-over.
2. Raster finishing with the same angle, a reduced step-over, and half the configured step-down.
- Pass flags: rough and finish enabled

Latency for successful runs is tracked in `TorchAI::lastLatencyMs()` to support quick smoke tests.

## Deployment Notes

- Set `TORCH_DIR` to the root of a LibTorch distribution (CPU or CUDA) before configuring CMake, or pass `-DTORCH_DIR=` explicitly. Toggle `-DWITH_TORCH=ON` to enable integration.
- When `WITH_TORCH` is off or LibTorch is not found, the project still builds and the AI layer remains ABI compatible, but stays in fallback mode.
- Model authors should verify outputs with the **Test Inference** button; the console prints the measured latency, predicted strategy, and device so regressions are easy to spot.
- Ship only vetted models. TorchScript and ONNX payloads can embed arbitrary compute graphs, so publish them from a trusted pipeline, record their checksums, and refuse to load unverified artefacts in production builds.
