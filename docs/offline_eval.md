# Offline Strategy Evaluation

The `tools/eval/run_eval.py` helper reproduces the desktop app's feature
extraction path so you can benchmark TorchScript or ONNX models entirely
offline. It works against local meshes and lightweight per-part metadata and
produces both detailed CSV outputs and a short Markdown report.

## Prerequisites

- Python 3.10 or newer.
- Python dependencies already used elsewhere in the repo:
  - `numpy`
  - `trimesh`
  - `onnxruntime` (only when using `.onnx` models)
  - `torch` (only when using TorchScript `.pt` models)

You can reuse the training environment by installing `train/requirements.txt`,
then add `onnxruntime` or `torch` as needed from your local package cache.

## Dataset Layout

Point the script at a directory containing one JSON file per part. Each JSON
entry references the mesh and provides the ground-truth metadata:

```json
{
  "id": "block_raster",
  "mesh": "block_raster.stl",
  "strategy": "raster",             // "raster" or "waterline"
  "raster_angle_deg": 45.0,
  "step_over_mm": 2.8,
  "user_step_over_mm": 2.5,
  "tool_diameter_mm": 6.0,
  "path_length_mm": 980.0,
  "feedrate_mm_per_min": 900.0
}
```

Meshes are assumed to live alongside the JSON (relative paths are supported).
`user_step_over_mm` is the value the operator provided; it is appended to the
feature vector alongside `tool_diameter_mm`.

The repository includes a tiny smoke set under `testdata/eval/smoke` for quick
sanity checks.

## Running the Evaluation

```bash
python tools/eval/run_eval.py \
  --dataset testdata/eval/smoke \
  --model models/test_run/strategy_v0.onnx \
  --output build/eval/smoke \
  --device auto
```

The script prints a short summary and writes:

- `metrics.csv` – per-sample predictions, absolute errors, logits, and timing proxy
- `report.md` – accuracy, macro-F1, confusion matrix, and aggregate statistics

The "time proxy" column/report value is the sum of `path_length_mm /
feedrate_mm_per_min` across the dataset (in minutes), which provides a rough
relative runtime indicator without needing the simulator.

Use `--device cpu` to pin inference to the CPU or `--device cuda` to require a
GPU execution provider (the script will exit early if the requested backend is
not available). If CUDA is requested but the system libraries (e.g.
`libcublasLt`) are missing, ONNX Runtime will log a failure for the CUDA
provider and transparently fall back to CPU execution.

## Notes

- Feature extraction mirrors `ai::FeatureExtractor` and pads/truncates according
  to the model card if present.
- TorchScript inference supports tuple, list, or dict outputs as emitted by the
  in-app models.
- ONNX inference relies on output names `logits`, `angle_deg`, and
  `step_over_mm`. If your model uses different names, update the script or
  export the `.onnx.json` descriptor to match.
